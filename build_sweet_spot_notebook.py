#!/usr/bin/env python3
"""Generate `ndt_sweet_spot_explorer.ipynb` for testing Multi-Node & Sweet-Spot Regimes."""

import json
import os


def make_md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def make_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def build_explorer_notebook() -> dict:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_sweet_spot_analysis.py"),
        "r",
        encoding="utf-8",
    ) as f:
        script_code = f.read()

    cells = [
        make_md(
            """# 🔎 NDT Sweet-Spot & Multi-Node Scale Explorer (II2202)
### When Does Decision-Model Policy Selection Actually Beat Static Policies & Simple Rules?

In our baseline 8-link experiment, all 8 links moved relatively homogeneously, so a simple single-threshold rule (`if max_delta > 6.5%: change-aware`) performed nearly as well as an AI Decision Model.

**This notebook explores the exact multi-node network conditions and topology scales (8, 16, 32, 48, 64 links) where simple rules break down and Decision-Model selection shines:**
1. **Scenario 1 — `localized_edge_noise` (32 Links across Core & Edge Nodes):**
   - 24 Core/Backbone links are calm except during two genuine network-wide routing waves (`t=[50..75s]` and `t=[120..145s]`).
   - Meanwhile, 2 non-critical Edge links constantly flicker with $\\pm 9\\%$ local jitter.
   - **What happens:** Because `change_aware` and `rule_switcher` trigger on `max_delta`, those 2 noisy edge links trick them into polling **all 32 links every 0.5s** for the entire 3 minutes (**182 KB / 343 polls**)!
   - **Why the Decision Model wins:** By reading the multi-link context (distinguishing 2 noisy edge links from a correlated 24-link core wave), the Decision Model ignores the false alarms and only activates `change-aware` during the two real core waves—cutting monitoring traffic by **58% (`75.5 KB` vs `182.5 KB`)** while reducing twin error by **31% (`2.19%` vs `3.20%` Fixed)**!
2. **Scenario 2 — `periodic_oscillation_trap` (32 Links):**
   - Zero-mean micro-oscillations on 3 links (`t=[20..110s]`) followed by a real network-wide burst storm (`t=[110..165s]`).
   - The Decision Model matches the `2.10%` accuracy of the rule switcher while saving **15% bandwidth** (`67.0 KB` vs `78.7 KB`).
3. **Scenario 3 — Topology Scale Sweep (`8 → 16 → 32 → 64 Links`):**
   - As network size grows from 8 links to 64 links, each poll message grows from `148 bytes` to `1,044 bytes`.
   - At 64 links, the Decision Model saves **172 KB of monitoring traffic per 3-minute window** compared to the Rule Switcher (`148 KB` vs `320 KB` — a **54% reduction**), while the model's own inference time (`~49 ms` on Laya) remains completely constant $O(1)$!"""
        ),
        make_code(script_code),
        make_md(
            """## 📊 Display Generated Sweet-Spot & Scale-Up Plots Inline"""
        ),
        make_code(
            """from IPython.display import Image, display
display(Image(filename="logs/sweet_spot_analysis/sweet_spot_fig1_multinode_comparison.png"))
display(Image(filename="logs/sweet_spot_analysis/sweet_spot_fig2_network_scale_sweep.png"))"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_explorer_notebook()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ndt_sweet_spot_explorer.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {out_path}")
