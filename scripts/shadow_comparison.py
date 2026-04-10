"""
scripts/shadow_comparison.py - v7 (primary) vs v8.1 (shadow) 2026 holdout比較

Usage:
    python scripts/shadow_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict_logic
from data_fetcher import get_multi_season_results
from scripts.backtest_runner import (
    run_walk_forward, rebuild_states, compute_ranks, build_elo, compute_metrics
)


def run_shadow_comparison(threshold_n: int = 80) -> dict:
    """
    v7 (primary) と v8.1 (shadow) を同じ2026データで評価し比較する。

    Returns
    -------
    {
        "v7": {metrics...},
        "v8_1": {metrics...},
        "n": int,
        "sufficient": bool,  # n >= threshold_n かどうか
    }
    """
    all_results = get_multi_season_results([2024, 2025], "j1")

    # v7 で2026評価
    res_v7 = run_walk_forward(all_results, eval_season=2026, predictor="integrated")

    # v8.1 shadow: MODEL_WEIGHTS と _3LOGIT_PARAMS を一時差し替えて評価
    orig_weights = dict(predict_logic.MODEL_WEIGHTS)
    orig_params = dict(predict_logic._3LOGIT_PARAMS)
    try:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(predict_logic.V8_1_MODEL_WEIGHTS)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(predict_logic.V8_1_3LOGIT_PARAMS)
        res_v8_1 = run_walk_forward(all_results, eval_season=2026, predictor="integrated")
    finally:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(orig_weights)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(orig_params)

    m_v7 = res_v7["metrics"]
    m_v8_1 = res_v8_1["metrics"]
    n = m_v7.get("n_samples", 0)

    return {
        "v7": m_v7,
        "v8_1": m_v8_1,
        "n": n,
        "sufficient": n >= threshold_n,
    }


def print_comparison(result: dict) -> None:
    print("=" * 70)
    print(f"  v7 (primary) vs v8.1 (shadow) - 2026 Holdout (n={result['n']})")
    print("=" * 70)
    print()

    metrics_to_show = [
        ("accuracy", "Accuracy", 4),
        ("f1_macro", "F1 macro", 4),
        ("log_loss", "Log Loss", 4),
        ("brier_score", "Brier",    4),
    ]
    print(f"{'Metric':<15} {'v7':>10} {'v8.1':>10} {'Diff':>10}")
    print("-" * 50)
    for key, label, dec in metrics_to_show:
        v7_val = result["v7"].get(key, 0)
        v8_val = result["v8_1"].get(key, 0)
        diff = v8_val - v7_val
        mark = ""
        if key in ("accuracy", "f1_macro"):
            mark = " (v8.1 better)" if diff > 0 else (" (v7 better)" if diff < 0 else "")
        else:  # log_loss, brier: lower is better
            mark = " (v8.1 better)" if diff < 0 else (" (v7 better)" if diff > 0 else "")
        print(f"{label:<15} {v7_val:>10.{dec}f} {v8_val:>10.{dec}f} {diff:+10.{dec}f}{mark}")

    print()
    v7_draw = result["v7"].get("predicted_distribution", {}).get("draw", 0)
    v8_draw = result["v8_1"].get("predicted_distribution", {}).get("draw", 0)
    print(f"{'Draw predicted':<15} {v7_draw:>10} {v8_draw:>10}")

    # 統計的信頼性判定
    print()
    if result["sufficient"]:
        print(f"n={result['n']} >= 80 → 統計的信頼性あり、採用判定可能")
    else:
        print(f"n={result['n']} < 80 → 統計的信頼性不十分、継続監視中")


if __name__ == "__main__":
    result = run_shadow_comparison()
    print_comparison(result)

    # レポート保存
    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "shadow_comparison_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")
