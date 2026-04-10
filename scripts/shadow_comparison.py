"""
scripts/shadow_comparison.py - Primary vs Baseline vs Shadow 2026 holdout比較

現在の構成:
- Primary: hybrid_v9.1 (v7 + Skellam dynamic 統合)
- Baseline: v7 refined (fallback)
- Shadow: v8.1 (内部ログ用)

Usage:
    python scripts/shadow_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict_logic
from data_fetcher import get_multi_season_results
from scripts.backtest_runner import (
    run_walk_forward, rebuild_states, compute_ranks, build_elo
)
from scripts.skellam_model import predict_skellam_dynamic
from scripts.calibration import predictions_to_arrays, compute_ece


LABELS = ["away", "draw", "home"]


def _metrics_from_probs(probs: np.ndarray, y_idx: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
    n = len(y_idx)
    if n == 0:
        return {}
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_idx, preds)
    f1 = f1_score(y_idx, preds, average="macro", zero_division=0)
    pm = np.clip(probs, 0.01, 0.99)
    pm = pm / pm.sum(axis=1, keepdims=True)
    ll = log_loss(y_idx, pm, labels=[0, 1, 2])
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(n), y_idx] = 1
    brier = float(np.mean(np.sum((pm - y_oh) ** 2, axis=1)))
    ece = compute_ece(pm, y_idx)
    pred_dist = {c: int((preds == i).sum()) for i, c in enumerate(LABELS)}
    cm = confusion_matrix(y_idx, preds, labels=[0, 1, 2])
    class_metrics = {}
    for i, c in enumerate(LABELS):
        tp = int(cm[i][i])
        fp = int(sum(cm[j][i] for j in range(3)) - tp)
        fn = int(sum(cm[i]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1c, 4)}
    return {
        "n": n, "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4), "brier": round(brier, 4), "ece": round(ece, 4),
        "pred_dist": pred_dist, "class_metrics": class_metrics,
    }


def _build_hybrid_preds(v7_preds: list, all_results: list) -> list[dict]:
    """v7予測からhybrid_v9.1予測を構築"""
    v7_map = {p["idx"]: p for p in v7_preds}
    hybrid_preds = []
    for idx, v7p in v7_map.items():
        match = all_results[idx]
        home, away = match["home"], match["away"]

        states = rebuild_states(all_results, idx)
        hs = states.get(home)
        as_ = states.get(away)
        if not hs or not as_:
            continue
        ranks = compute_ranks(states)
        h_stats = hs.to_stats_dict(ranks.get(home, 99))
        a_stats = as_.to_stats_dict(ranks.get(away, 99))

        elo = build_elo(all_results, idx, k=32.0, home_bonus=50.0)
        elo_h, elo_a = elo.score_pair(home, away)

        # v7予測を確率%に変換
        v7_pred = {
            "home_win_prob": round(v7p["prob_home"] * 100),
            "draw_prob": round(v7p["prob_draw"] * 100),
            "away_win_prob": round(v7p["prob_away"] * 100),
        }
        hyb = predict_logic.compute_hybrid_v9(
            home, away, h_stats, a_stats,
            hs.get_form(5), as_.get_form(5),
            v7_prediction=v7_pred,
            elo_home_score=elo_h, elo_away_score=elo_a,
        )
        hybrid_preds.append({
            "idx": idx,
            "actual": v7p["actual"],
            "prob_home": hyb["home_win_prob"] / 100,
            "prob_draw": hyb["draw_prob"] / 100,
            "prob_away": hyb["away_win_prob"] / 100,
            "selection": hyb["selection"],
        })
    return hybrid_preds


def _preds_to_arrays(preds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not preds:
        return np.array([]), np.array([])
    probs = np.array([[p["prob_away"], p["prob_draw"], p["prob_home"]] for p in preds])
    y_idx = np.array([{"away": 0, "draw": 1, "home": 2}[p["actual"]] for p in preds])
    return probs, y_idx


def run_comparison(threshold_n: int = 80) -> dict:
    """primary / baseline / shadow の比較"""
    all_results = get_multi_season_results([2024, 2025], "j1")

    result = {"models": {}, "threshold_n": threshold_n}

    # Baseline: v7 で2026評価
    res_v7 = run_walk_forward(all_results, eval_season=2026, predictor="integrated")
    probs_v7, y_v7 = _preds_to_arrays(res_v7["predictions"])
    result["models"]["v7_baseline"] = _metrics_from_probs(probs_v7, y_v7)

    # Primary: hybrid_v9.1 (v7予測から構築)
    hybrid_preds = _build_hybrid_preds(res_v7["predictions"], all_results)
    probs_hyb, y_hyb = _preds_to_arrays(hybrid_preds)
    hyb_metrics = _metrics_from_probs(probs_hyb, y_hyb)
    # 選択ログ集計
    selection_log = {"v7": 0, "skellam": 0, "weighted": 0}
    for p in hybrid_preds:
        selection_log[p["selection"]] = selection_log.get(p["selection"], 0) + 1
    hyb_metrics["selection_log"] = selection_log
    result["models"]["hybrid_v9.1_primary"] = hyb_metrics

    # Shadow: v8.1
    orig_w = dict(predict_logic.MODEL_WEIGHTS)
    orig_p = dict(predict_logic._3LOGIT_PARAMS)
    try:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(predict_logic.V8_1_MODEL_WEIGHTS)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(predict_logic.V8_1_3LOGIT_PARAMS)
        res_v81 = run_walk_forward(all_results, eval_season=2026, predictor="integrated")
    finally:
        predict_logic.MODEL_WEIGHTS.clear()
        predict_logic.MODEL_WEIGHTS.update(orig_w)
        predict_logic._3LOGIT_PARAMS.clear()
        predict_logic._3LOGIT_PARAMS.update(orig_p)
    probs_v81, y_v81 = _preds_to_arrays(res_v81["predictions"])
    result["models"]["v8.1_shadow"] = _metrics_from_probs(probs_v81, y_v81)

    # n 判定
    n = hyb_metrics.get("n", 0)
    result["n"] = n
    result["sufficient"] = n >= threshold_n
    return result


def print_report(result: dict) -> None:
    print("=" * 78)
    print("  Primary (hybrid_v9.1) vs Baseline (v7) vs Shadow (v8.1) - 2026 Holdout")
    print("=" * 78)
    print()
    print(f"{'Model':<22} {'n':>4} {'Acc':>7} {'F1':>7} {'LogL':>7} {'Brier':>7} {'ECE':>6} {'Draw#':>6}")
    print("-" * 78)
    for name in ["hybrid_v9.1_primary", "v7_baseline", "v8.1_shadow"]:
        m = result["models"].get(name, {})
        if not m:
            continue
        print(f"{name:<22} {m.get('n',0):>4} {m.get('accuracy',0):>7.4f} {m.get('f1_macro',0):>7.4f} "
              f"{m.get('log_loss',0):>7.4f} {m.get('brier',0):>7.4f} {m.get('ece',0):>6.4f} "
              f"{m.get('pred_dist',{}).get('draw',0):>6}")

    # 選択ログ
    hyb = result["models"].get("hybrid_v9.1_primary", {})
    sl = hyb.get("selection_log", {})
    if sl:
        print(f"\nhybrid_v9.1 selection: v7={sl.get('v7',0)} skellam={sl.get('skellam',0)} weighted={sl.get('weighted',0)}")

    # 統計的信頼性
    print()
    n = result.get("n", 0)
    if result.get("sufficient"):
        print(f"n={n} >= {result['threshold_n']} → 統計的信頼性あり、正式定着判定可能")
    else:
        print(f"n={n} < {result['threshold_n']} → 統計的信頼性不十分、継続監視中")


if __name__ == "__main__":
    result = run_comparison()
    print_report(result)

    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "holdout_2026_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")
