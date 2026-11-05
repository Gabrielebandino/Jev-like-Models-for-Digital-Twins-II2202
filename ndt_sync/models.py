"""Unified Model Adapters and Sequential T4 GPU Server Managers for System-One Decision Models.

Supports the three open-weight candidate models in sequential isolation mode:
1. `laya` (`convaiinnovations/laya` via `laya.serve` on port 8000)
2. `kev_4b` (`jaredpalmer/kev-4b` via `kev.serve` with `KEV_DTYPE=fp16` on port 8008)
3. `clm_8b` (`Contrastive-LM/CLM-v0.1-8B` via lightweight 4-bit NF4 `Qwen/Qwen3-8B`
   last-token pooling embedder + native `clm` InfoNCE projection heads on port 8700)

All three expose the identical TypeSafe-compatible `POST /v1/systemone` API contract
and are queried through a single unified `SystemOneHTTPSelector` client.
"""

from __future__ import annotations
import os
import sys
import gc
import time
import json
import signal
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests
import numpy as np

from .selectors import (
    ReviewSummary,
    SelectorDecisionResult,
    FROZEN_POLICY_CHOICE_QUESTION,
)


class SystemOneHTTPSelector:
    """Unified HTTP client querying any `/v1/systemone` endpoint (Laya, Kev-4B, CLM-v0.1-8B, or Jev)."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_model_field: str = "laya",
        api_key: str = "sk-local-kaggle-token",
        timeout_s: float = 5.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/systemone"
        self.api_model_field = api_model_field
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def select_policy(self, summary: ReviewSummary) -> SelectorDecisionResult:
        """Query `/v1/systemone` with the frozen two-option Choice schema and measure exact RTT latency."""
        payload = {
            "model": self.api_model_field,
            "state": summary.to_prompt_text(),
            "questions": FROZEN_POLICY_CHOICE_QUESTION,
        }
        payload_str = json.dumps(payload, ensure_ascii=False)
        req_bytes = len(payload_str.encode("utf-8"))

        t0 = time.perf_counter()
        try:
            resp = self.session.post(self.endpoint, data=payload_str.encode("utf-8"), timeout=self.timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rep_bytes = len(resp.content)
            raw_text = resp.text

            if resp.status_code != 200:
                return SelectorDecisionResult(
                    selected_policy=None,
                    decision_latency_ms=latency_ms,
                    probabilities={},
                    confidence=0.0,
                    request_bytes=req_bytes,
                    reply_bytes=rep_bytes,
                    raw_output=raw_text[:500],
                    is_valid=False,
                    error_reason=f"http_{resp.status_code}",
                )

            data = resp.json()
            ans = (data.get("answers") or {}).get("selected_policy") or {}
            choice = ans.get("choice")
            probs = ans.get("probabilities") or {}
            conf = float(ans.get("confidence", ans.get("answer_confidence", 0.0)) or 0.0)

            # Normalize probabilities
            p_fixed = float(probs.get("fixed", 0.0))
            p_ca = float(probs.get("change-aware", probs.get("change_aware", 0.0)))
            if choice == "change_aware":
                choice = "change-aware"

            is_valid = choice in ("fixed", "change-aware")
            return SelectorDecisionResult(
                selected_policy=choice if is_valid else None,
                decision_latency_ms=latency_ms,
                probabilities={"fixed": p_fixed, "change-aware": p_ca},
                confidence=conf,
                request_bytes=req_bytes,
                reply_bytes=rep_bytes,
                raw_output=raw_text[:600],
                is_valid=is_valid,
                error_reason=None if is_valid else f"unrecognized_choice:{choice!r}",
            )
        except requests.Timeout:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SelectorDecisionResult(
                selected_policy=None,
                decision_latency_ms=latency_ms,
                probabilities={},
                confidence=0.0,
                request_bytes=req_bytes,
                reply_bytes=0,
                raw_output="TIMEOUT",
                is_valid=False,
                error_reason="request_timeout",
            )
        except Exception as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SelectorDecisionResult(
                selected_policy=None,
                decision_latency_ms=latency_ms,
                probabilities={},
                confidence=0.0,
                request_bytes=req_bytes,
                reply_bytes=0,
                raw_output=str(ex)[:300],
                is_valid=False,
                error_reason=f"exception:{type(ex).__name__}",
            )


def get_gpu_vram_usage_mb() -> Dict[str, Any]:
    """Query current NVIDIA GPU memory allocated / used via nvidia-smi or torch.cuda."""
    info: Dict[str, Any] = {"gpus": []}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                info["gpus"].append(
                    {
                        "gpu_index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "gpu_util_pct": float(parts[4]),
                    }
                )
    except Exception:
        pass
    return info


# ==============================================================================
# STANDALONE DIRECT PYTORCH 4-BIT SERVER SCRIPT FOR CLM-v0.1-8B (NO VLLM NEEDED)
# ==============================================================================
CLM_TRANSFORMERS_SERVER_CODE = '''"""Standalone FastAPI Server for Contrastive-LM/CLM-v0.1-8B using 4-bit Transformers (No vLLM).

Loads `Qwen/Qwen3-8B` in 4-bit NF4 (`float16` compute dtype for NVIDIA T4 Turing GPUs)
with last-token pooling and pairs it with the official `CLM_v0.1-8B.pt` InfoNCE state/action
projection heads from `Contrastive-LM/CLM-v0.1-8B`, exposing `POST /v1/systemone` and `GET /health`.
"""
import os
import math
import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

PORT = int(os.environ.get("CLM_PORT", "8700"))
BASE_MODEL = os.environ.get("CLM_BASE_MODEL", "Qwen/Qwen3-8B")
CLM_REPO = os.environ.get("CLM_REPO", "Contrastive-LM/CLM-v0.1-8B")
CLM_FILE = os.environ.get("CLM_FILE", "CLM_v0.1-8B.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[CLM-Server] Downloading projection head {CLM_REPO}/{CLM_FILE}...")
ckpt_path = hf_hub_download(repo_id=CLM_REPO, filename=CLM_FILE)
ck = torch.load(ckpt_path, map_location="cpu")
cfg = dict(ck["cfg"])

def make_head(width: int, depth: int = 2, proj: int = 512, activation: str = "gelu",
              layernorm: bool = False, residual: bool = False, hidden: int = 4096):
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual
        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)
    return Head()

kw = dict(
    width=cfg["width"],
    depth=cfg["depth"],
    proj=ck.get("projection_dim", cfg.get("projection_dim", 512)),
    activation=cfg.get("activation", "gelu"),
    layernorm=cfg.get("layernorm", False),
    residual=cfg.get("residual", False),
    hidden=cfg.get("hidden_size", 4096),
)
state_head = make_head(**kw)
action_head = make_head(**kw)
state_head.load_state_dict(ck["state_head"])
action_head.load_state_dict(ck["action_head"])
state_head.eval().to(DEVICE)
action_head.eval().to(DEVICE)
logit_scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

print(f"[CLM-Server] Loading 4-bit NF4 encoder {BASE_MODEL} onto {DEVICE}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
encoder = AutoModel.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0} if torch.cuda.is_available() else "auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
encoder.eval()

# Action & State embedding LRU cache
EMB_CACHE: Dict[str, torch.Tensor] = {}

@torch.no_grad()
def encode_texts_last_token(texts: List[str]) -> torch.Tensor:
    missing = [t for t in texts if t not in EMB_CACHE]
    if missing:
        enc = tokenizer(
            missing,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        out = encoder(**enc)
        hidden = out.last_hidden_state  # [B, L, D]
        # Last-token pooling using attention_mask
        seq_lengths = enc["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=DEVICE)
        pooled = hidden[batch_indices, seq_lengths].float()
        pooled = F.normalize(pooled, p=2, dim=-1)
        for idx, t_str in enumerate(missing):
            if len(EMB_CACHE) > 4096:
                EMB_CACHE.pop(next(iter(EMB_CACHE)))
            EMB_CACHE[t_str] = pooled[idx]
    return torch.stack([EMB_CACHE[t] for t in texts], dim=0)

app = FastAPI(title="CLM-v0.1-8B SystemOne Server")

@app.get("/health")
def health():
    return {"status": "ok", "model": "Contrastive-LM/CLM-v0.1-8B", "encoder": BASE_MODEL, "device": DEVICE}

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "clm-latest"}, {"id": "CLM-v0.1-8B"}]}

@app.post("/v1/systemone")
@torch.no_grad()
def systemone(body: Dict[str, Any]):
    state_raw = body.get("state", "")
    state_str = state_raw if isinstance(state_raw, str) else str(state_raw)
    questions = body.get("questions", {})
    answers: Dict[str, Any] = {}
    total_input_tokens = len(tokenizer.encode(state_str))

    for qid, qspec in questions.items():
        qtype = qspec.get("type", "choice")
        ins = str(qspec.get("instructions", "")).strip()
        crit = qspec.get("criteria", {})

        full_state_text = f"{state_str}\\n\\n{ins}" if ins else state_str
        if qtype == "choice":
            keys = list(crit.keys())
            cand_texts = [str(crit[k]) if crit[k] not in (None, "") else str(k) for k in keys]
        elif qtype == "noul":
            keys = ["false", "true"]
            cand_texts = [
                f"false: No. This is false: {ins}",
                f"true: Yes. This is true: {ins}",
            ]
        else:
            keys = [str(i) for i in range(len(crit))]
            cand_texts = [str(c) for c in crit]

        s_emb = encode_texts_last_token([full_state_text])  # [1, 4096]
        c_emb = encode_texts_last_token(cand_texts)         # [K, 4096]

        s_proj = F.normalize(state_head(s_emb), p=2, dim=-1)   # [1, 512]
        c_proj = F.normalize(action_head(c_emb), p=2, dim=-1)  # [K, 512]

        logits = (s_proj @ c_proj.T).squeeze(0) * logit_scale
        probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_map = {k: round(float(p), 6) for k, p in zip(keys, probs_t)}

        if qtype == "choice":
            best_idx = int(np.argmax(probs_t))
            best_key = keys[best_idx]
            sorted_p = np.sort(probs_t)[::-1]
            conf = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
            answers[qid] = {
                "type": "choice",
                "choice": best_key,
                "probabilities": probs_map,
                "confidence": round(conf, 4),
            }
        elif qtype == "noul":
            p_true = float(probs_map.get("true", 0.5))
            answers[qid] = {
                "type": "noul",
                "noul": round(p_true, 6),
                "probabilities": probs_map,
                "confidence": round(abs(2.0 * p_true - 1.0), 4),
            }

    return {
        "model": "CLM-v0.1-8B",
        "answers": answers,
        "usage": {"input_tokens": total_input_tokens, "output_tokens": 0},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
'''
