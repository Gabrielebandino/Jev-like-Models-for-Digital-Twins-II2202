#!/usr/bin/env python3
"""Unified Live GPU Model Servers & Clients for NDT v2 Experiments (Kaggle T4 x2).

Supports 3 distinct open-weight decision-model families in strict sequential isolation (zero VRAM contention):
  1. `laya`      : convaiinnovations/laya (421M ModernBERT-large System-1 Choice, fp16, ~48 ms)
  2. `clm_8b`    : Contrastive-LM/CLM-v0.1-8B (4-bit NF4 Qwen3-8B + InfoNCE projection heads, ~550 ms)
  3. `qwen25_7b` : Qwen/Qwen2.5-7B-Instruct (4-bit NF4, single-pass exact A/B logit scoring, ~200 ms)
                   Tested in BOTH `qwen25_7b_zeroshot` and training-free `qwen25_7b_fewshot`
                   (4 oracle-labeled pilot examples in-context, zero weight updates).

All models implement:
  - Strictly NEUTRAL telemetry prompts (zero if/else spoon-feeding)
  - 4 pre-registered prompt paraphrases (including option-order swap)
  - Zhao et al. (ICML 2021) training-free Contextual Calibration on content-free baseline input
  - Exact RTT wall-clock latency measurement, request/reply byte accounting, and VRAM logging
"""

from __future__ import annotations
import os
import sys
import gc
import json
import time
import math
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from ndt_v2_engine import (
    ProtocolV2,
    ReviewFeatures,
    build_neutral_prompt_paraphrases,
    format_neutral_telemetry_text,
    build_content_free_telemetry_text,
    apply_contextual_calibration,
)


def get_gpu_vram_mb() -> Dict[str, Any]:
    """Query current NVIDIA GPU memory footprint via nvidia-smi."""
    info: Dict[str, Any] = {"gpus": []}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                info["gpus"].append({
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "gpu_util_pct": float(parts[4]),
                })
    except Exception:
        pass
    return info


def kill_all_model_servers() -> None:
    for pat in ("laya.serve", "clm_v2_server.py"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(2)


def wait_for_health(url: str, timeout_s: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ==============================================================================
# STANDALONE 4-BIT FASTAPI SERVER FOR CLM-v0.1-8B (NO VLLM CONFLICTS ON T4)
# ==============================================================================
CLM_V2_SERVER_SCRIPT = '''"""Standalone 4-bit NF4 FastAPI Server for Contrastive-LM/CLM-v0.1-8B on Kaggle T4."""
import os
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

PORT = int(os.environ.get("CLM_PORT", "8700"))
BASE_MODEL = os.environ.get("CLM_BASE_MODEL", "Qwen/Qwen3-8B")
CLM_REPO = os.environ.get("CLM_REPO", "Contrastive-LM/CLM-v0.1-8B")
CLM_FILE = os.environ.get("CLM_FILE", "CLM_v0.1-8B.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ckpt_path = hf_hub_download(repo_id=CLM_REPO, filename=CLM_FILE)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"])

def make_head(width, depth=2, proj=512, activation="gelu", layernorm=False, residual=False, hidden=4096):
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
    width=cfg["width"], depth=cfg["depth"],
    proj=ck.get("projection_dim", cfg.get("projection_dim", 512)),
    activation=cfg.get("activation", "gelu"),
    layernorm=cfg.get("layernorm", False),
    residual=cfg.get("residual", False),
    hidden=cfg.get("hidden_size", 4096),
)
state_head = make_head(**kw).eval().to(DEVICE)
action_head = make_head(**kw).eval().to(DEVICE)
state_head.load_state_dict(ck["state_head"])
action_head.load_state_dict(ck["action_head"])
logit_scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
encoder = AutoModel.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config,
    device_map={"": 0} if torch.cuda.is_available() else "auto",
    torch_dtype=torch.float16, trust_remote_code=True,
).eval()

EMB_CACHE: Dict[str, torch.Tensor] = {}

@torch.no_grad()
def encode_last_token(texts: List[str]) -> torch.Tensor:
    missing = [t for t in texts if t not in EMB_CACHE]
    if missing:
        enc = tokenizer(missing, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
        hidden = encoder(**enc).last_hidden_state
        seq_len = enc["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=DEVICE), seq_len].float()
        pooled = F.normalize(pooled, p=2, dim=-1)
        for idx, ts in enumerate(missing):
            if len(EMB_CACHE) > 2048:
                EMB_CACHE.pop(next(iter(EMB_CACHE)))
            EMB_CACHE[ts] = pooled[idx]
    return torch.stack([EMB_CACHE[t] for t in texts], dim=0)

app = FastAPI(title="CLM-v0.1-8B v2 Server")

@app.get("/health")
def health():
    return {"status": "ok", "model": "CLM-v0.1-8B", "device": DEVICE}

@app.post("/v1/systemone")
@torch.no_grad()
def systemone(body: Dict[str, Any]):
    state_str = str(body.get("state", ""))
    questions = body.get("questions", {})
    answers = {}
    for qid, qspec in questions.items():
        ins = str(qspec.get("instructions", "")).strip()
        crit = qspec.get("criteria", {})
        full_state = f"{state_str}\\n\\n{ins}" if ins else state_str
        keys = list(crit.keys())
        cands = [str(crit[k]) for k in keys]
        s_proj = F.normalize(state_head(encode_last_token([full_state])), p=2, dim=-1)
        c_proj = F.normalize(action_head(encode_last_token(cands)), p=2, dim=-1)
        logits = (s_proj @ c_proj.T).squeeze(0) * logit_scale
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pmap = {k: round(float(p), 6) for k, p in zip(keys, probs)}
        best_k = keys[int(np.argmax(probs))]
        answers[qid] = {"type": "choice", "choice": best_k, "probabilities": pmap}
    return {"model": "CLM-v0.1-8B", "answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
'''


# ==============================================================================
# SYSTEM-ONE HTTP SELECTOR (LAYA & CLM-v0.1-8B)
# ==============================================================================
class SystemOneV2Client:
    """Queries live `/v1/systemone` endpoint (`laya` or `clm_8b`) with optional Zhao calibration."""

    def __init__(
        self,
        model_key: str,
        base_url: str,
        api_model_field: str,
        paraphrase: Dict[str, Any],
        use_calibration: bool = True,
        prior_p_ca: float = 0.5,
        timeout_s: float = 5.0,
    ):
        self.model_key = model_key
        self.endpoint = f"{base_url.rstrip('/')}/v1/systemone"
        self.api_model_field = api_model_field
        self.paraphrase = paraphrase
        self.use_calibration = use_calibration
        self.prior_p_ca = prior_p_ca
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": "Bearer sk-local-kaggle-token",
            "Content-Type": "application/json",
        })

    def query_text(self, state_text: str) -> Dict[str, Any]:
        q_spec = {
            "selected_policy": {
                "type": "choice",
                "instructions": self.paraphrase["instructions"],
                "criteria": self.paraphrase["criteria"],
            }
        }
        payload = {
            "model": self.api_model_field,
            "state": state_text,
            "questions": q_spec,
        }
        raw_req = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        t0 = time.perf_counter()
        try:
            resp = self.session.post(self.endpoint, data=raw_req, timeout=self.timeout_s)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                return {
                    "choice": None, "is_valid": False, "latency_ms": lat_ms,
                    "p_ca_raw": 0.5, "p_ca_cal": 0.5, "req_bytes": len(raw_req),
                    "rep_bytes": len(resp.content), "error": f"http_{resp.status_code}",
                    "raw_output": resp.text[:250],
                }
            data = resp.json()
            ans = (data.get("answers") or {}).get("selected_policy") or {}
            choice_raw = ans.get("choice")
            if choice_raw == "change_aware":
                choice_raw = "change-aware"
            probs = ans.get("probabilities") or {}
            p_fix = float(probs.get("fixed", 0.0))
            p_ca = float(probs.get("change-aware", probs.get("change_aware", 0.0)))
            tot = p_fix + p_ca
            p_ca_norm = (p_ca / tot) if tot > 1e-9 else (1.0 if choice_raw == "change-aware" else 0.0)
            is_valid = choice_raw in ("fixed", "change-aware")
            if self.use_calibration and is_valid:
                choice_final, p_cal = apply_contextual_calibration(p_ca_norm, self.prior_p_ca)
            else:
                choice_final, p_cal = choice_raw, p_ca_norm
            return {
                "choice": choice_final if is_valid else None,
                "choice_raw": choice_raw,
                "is_valid": is_valid,
                "latency_ms": lat_ms,
                "p_ca_raw": p_ca_norm,
                "p_ca_cal": p_cal,
                "req_bytes": len(raw_req),
                "rep_bytes": len(resp.content),
                "error": None if is_valid else f"invalid_choice:{choice_raw!r}",
                "raw_output": resp.text[:250],
            }
        except Exception as ex:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "choice": None, "is_valid": False, "latency_ms": lat_ms,
                "p_ca_raw": 0.5, "p_ca_cal": 0.5, "req_bytes": len(raw_req),
                "rep_bytes": 0, "error": f"{type(ex).__name__}:{ex}", "raw_output": str(ex)[:200],
            }

    def __call__(self, feat: ReviewFeatures) -> Dict[str, Any]:
        return self.query_text(format_neutral_telemetry_text(feat))


# ==============================================================================
# GENERATIVE 7B INSTRUCT LOGIT-SCORED SELECTOR (ZERO-SHOT & PILOT FEW-SHOT)
# ==============================================================================
class Qwen25LogitSelector:
    """In-process 4-bit NF4 Qwen2.5-7B-Instruct selector via exact single-pass A/B logit scoring.

    Advantages:
      - Zero parse failure (exact softmax over single-token logits for 'A' vs 'B')
      - Single forward pass (~150-280 ms on T4, well under 2000 ms deadline)
      - Exact continuous probabilities for ROC-AUC & Zhao et al. (2021) contextual calibration
      - Supports training-free few-shot in-context examples drawn strictly from Pilot oracle reviews
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map={"": 0} if torch.cuda.is_available() else "auto",
            torch_dtype=torch.float16, trust_remote_code=True,
        ).eval()
        # Resolve exact token IDs for "A" and "B"
        self.tok_A = self.tokenizer.encode("A", add_special_tokens=False)[-1]
        self.tok_B = self.tokenizer.encode("B", add_special_tokens=False)[-1]

    def make_callable(
        self,
        paraphrase: Dict[str, Any],
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
        use_calibration: bool = True,
        prior_p_ca: float = 0.5,
    ):
        opt0, opt1 = paraphrase["option_order"][0], paraphrase["option_order"][1]
        c0 = paraphrase["criteria"][opt0]
        c1 = paraphrase["criteria"][opt1]

        sys_prompt = (
            f"{paraphrase['instructions']}\n\n"
            f"Options:\n"
            f"A) {opt0}: {c0}\n"
            f"B) {opt1}: {c1}\n\n"
            f"Respond with a single letter: A or B."
        )
        fs_block = ""
        if few_shot_examples:
            parts = ["Here are pilot examples from previous traces showing the better-performing policy:"]
            for idx_ex, (ex_txt, ex_lbl) in enumerate(few_shot_examples, 1):
                letter = "A" if ex_lbl == opt0 else "B"
                parts.append(f"Example {idx_ex}:\n{ex_txt}\nAnswer: {letter}")
            fs_block = "\n\n".join(parts) + "\n\nNow decide for the current review:\n"

        def query_text(state_text: str) -> Dict[str, Any]:
            import torch
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{fs_block}{state_text}\nAnswer:"},
            ]
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            req_bytes = len(prompt_str.encode("utf-8"))
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    enc = self.tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
                    logits = self.model(**enc).logits[0, -1, :]
                    la = float(logits[self.tok_A].item())
                    lb = float(logits[self.tok_B].item())
                    m = max(la, lb)
                    pa = math.exp(la - m) / (math.exp(la - m) + math.exp(lb - m))
                    pb = 1.0 - pa
                lat_ms = (time.perf_counter() - t0) * 1000.0
                p_ca_raw = pa if opt0 == "change-aware" else pb
                choice_raw = "change-aware" if p_ca_raw >= 0.5 else "fixed"
                if use_calibration:
                    choice_final, p_cal = apply_contextual_calibration(p_ca_raw, prior_p_ca)
                else:
                    choice_final, p_cal = choice_raw, p_ca_raw
                return {
                    "choice": choice_final,
                    "choice_raw": choice_raw,
                    "is_valid": True,
                    "latency_ms": lat_ms,
                    "p_ca_raw": round(p_ca_raw, 6),
                    "p_ca_cal": round(p_cal, 6),
                    "req_bytes": req_bytes,
                    "rep_bytes": 4,
                    "error": None,
                    "raw_output": f"logitA={la:.2f},logitB={lb:.2f},pA={pa:.4f},pB={pb:.4f}",
                }
            except Exception as ex:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "choice": None, "is_valid": False, "latency_ms": lat_ms,
                    "p_ca_raw": 0.5, "p_ca_cal": 0.5, "req_bytes": req_bytes,
                    "rep_bytes": 0, "error": f"{type(ex).__name__}:{ex}", "raw_output": str(ex)[:200],
                }

        def selector_fn(feat: ReviewFeatures) -> Dict[str, Any]:
            return query_text(format_neutral_telemetry_text(feat))

        selector_fn.query_text = query_text  # type: ignore[attr-defined]
        return selector_fn

    def unload(self) -> None:
        import torch
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# SERVER LIFECYCLE HELPERS
# ==============================================================================
def start_http_model_server(model_key: str, log_dir: str = "logs/v2") -> Tuple[subprocess.Popen, str, str]:
    """Start `laya` or `clm_8b` server cleanly on GPU 0 and verify `/health`."""
    os.makedirs(log_dir, exist_ok=True)
    kill_all_model_servers()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if model_key == "laya":
        port = 8000
        env["LAYA_HOST"] = "127.0.0.1"
        env["LAYA_PORT"] = str(port)
        env["LAYA_MODELS"] = "english"
        env["LAYA_DEVICE"] = "cuda"
        log_path = os.path.join(log_dir, "server_laya.log")
        lf = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "laya.serve"],
            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        api_field = "laya"
    elif model_key == "clm_8b":
        port = 8700
        env["CLM_PORT"] = str(port)
        with open("clm_v2_server.py", "w", encoding="utf-8") as f:
            f.write(CLM_V2_SERVER_SCRIPT)
        log_path = os.path.join(log_dir, "server_clm_8b.log")
        lf = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "clm_v2_server.py"],
            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        api_field = "CLM-v0.1-8B"
    else:
        raise ValueError(f"Unsupported HTTP server model_key: {model_key}")

    print(f"[{model_key}] Starting server on {base_url} (PID={proc.pid})...")
    if not wait_for_health(f"{base_url}/health", timeout_s=240):
        tail = open(log_path).read()[-2500:] if os.path.exists(log_path) else ""
        raise RuntimeError(f"[{model_key}] Server failed to become healthy!\n{tail}")
    print(f"✓ [{model_key}] Server ONLINE ({base_url})")
    return proc, base_url, api_field


def calibrate_and_select_paraphrase_on_pilot(
    model_key: str,
    pilot_review_items: List[Tuple[str, str, float]],  # (telemetry_text, oracle_label, oracle_margin)
    paraphrases: List[Dict[str, Any]],
    build_query_fn: Any,  # fn(paraphrase) -> fn(text) -> dict
    num_links: int = 8,
) -> Dict[str, Any]:
    """Screen all 4 neutral paraphrases + Zhao et al. (2021) contextual calibration strictly on Pilot reviews.

    Returns frozen `best_paraphrase_idx`, content-free `prior_p_ca`, pilot `roc_auc`, and paraphrase spread!
    """
    from sklearn.metrics import roc_auc_score

    cf_text = build_content_free_telemetry_text(num_links=num_links)
    y_true = np.array([1 if lbl == "change-aware" else 0 for _, lbl, _ in pilot_review_items], dtype=np.int32)
    screen_rows = []

    for idx, para in enumerate(paraphrases):
        qfn = build_query_fn(para)
        # 1. Measure content-free prior bias p0(change-aware) per Zhao et al. (ICML 2021)
        cf_res = qfn(cf_text)
        p0_ca = float(cf_res.get("p_ca_raw", 0.5))

        # 2. Score pilot reviews
        p_raw_list, lats, valids = [], [], []
        for txt, _, _ in pilot_review_items:
            res = qfn(txt)
            p_raw_list.append(float(res.get("p_ca_raw", 0.5)))
            lats.append(float(res.get("latency_ms", 0.0)))
            valids.append(bool(res.get("is_valid", False)))

        p_raw = np.array(p_raw_list, dtype=np.float64)
        # Apply Zhao et al. calibration using content-free prior (and fallback to pilot median if extreme)
        p_cal = np.array([apply_contextual_calibration(p, p0_ca)[1] for p in p_raw])
        # Also compute pilot-quantile calibrated prior (training-free label-free centering at pilot median!)
        p_med_prior = float(np.clip(np.median(p_raw), 0.02, 0.98))
        p_cal_med = np.array([apply_contextual_calibration(p, p_med_prior)[1] for p in p_raw])

        auc = float(roc_auc_score(y_true, p_raw)) if len(np.unique(y_true)) == 2 else 0.5
        acc_raw = float(np.mean((p_raw >= 0.5) == y_true))
        acc_cal_cf = float(np.mean((p_cal >= 0.5) == y_true))
        acc_cal_med = float(np.mean((p_cal_med >= 0.5) == y_true))

        # Choose better of content-free prior vs label-free pilot-median prior strictly on pilot
        use_prior = p0_ca if acc_cal_cf >= acc_cal_med else p_med_prior
        prior_method = "zhao_content_free" if acc_cal_cf >= acc_cal_med else "label_free_pilot_median"
        acc_cal_best = max(acc_cal_cf, acc_cal_med)

        screen_rows.append({
            "model_key": model_key,
            "paraphrase_idx": idx,
            "paraphrase_name": para["name"],
            "parse_valid_rate": round(float(np.mean(valids)), 4),
            "latency_p50_ms": round(float(np.median(lats)), 2),
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "content_free_p0_ca": round(p0_ca, 4),
            "pilot_median_p_ca": round(p_med_prior, 4),
            "selected_prior_p_ca": round(use_prior, 4),
            "prior_method": prior_method,
            "pilot_roc_auc": round(auc, 4),
            "pilot_acc_raw": round(acc_raw, 4),
            "pilot_acc_calibrated": round(acc_cal_best, 4),
            "pilot_raw_ca_frac": round(float(np.mean(p_raw >= 0.5)), 4),
        })

    # Pick paraphrase with highest pilot ROC-AUC (tie-break by calibrated accuracy)
    best_row = max(screen_rows, key=lambda r: (r["pilot_roc_auc"], r["pilot_acc_calibrated"]))
    aucs = [r["pilot_roc_auc"] for r in screen_rows]
    return {
        "model_key": model_key,
        "best_paraphrase_idx": best_row["paraphrase_idx"],
        "best_paraphrase_name": best_row["paraphrase_name"],
        "selected_prior_p_ca": best_row["selected_prior_p_ca"],
        "prior_method": best_row["prior_method"],
        "best_pilot_roc_auc": best_row["pilot_roc_auc"],
        "paraphrase_auc_spread": round(float(max(aucs) - min(aucs)), 4),
        "best_latency_p95_ms": best_row["latency_p95_ms"],
        "parse_valid_rate": best_row["parse_valid_rate"],
        "all_paraphrases": screen_rows,
    }
