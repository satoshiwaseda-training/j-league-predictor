"""
scripts/monitoring.py - Primary モデル監視レポート

prediction_store.json から直近20/40試合の primary vs baseline vs shadow 比較を
ローリングウィンドウで実行し、自動降格判定・早期警戒レベルを出力する。

Usage:
    python scripts/monitoring.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from prediction_store import load_all
from scripts.monitoring_rules import (
    ROLLING_WINDOWS,
    evaluate_warning_level,
    should_auto_downgrade,
    MIN_SAMPLES_FOR_WARNING,
    MIN_SAMPLES_FOR_DOWNGRADE,
    WARNING_LEVELS,
    PREDICTION_SCHEMA_VERSION,
)


LABELS = ["away", "draw", "home"]


def _metrics(preds: list[dict], side_key: str = "prediction") -> dict:
    """指定されたside (prediction/baseline_prediction/shadow_prediction) のmetrics"""
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    n = 0
    y_true = []
    y_pred = []
    y_probs = []
    for p in preds:
        side = p.get(side_key)
        if not side:
            continue
        actual = (p.get("actual") or {}).get("winner")
        if not actual:
            continue
        h = int(side.get("home_win_prob", 40))
        d = int(side.get("draw_prob", 25))
        a = int(side.get("away_win_prob", 35))
        if h >= a and h >= d:
            pw = "home"
        elif a > h and a >= d:
            pw = "away"
        else:
            pw = "draw"
        y_true.append(actual)
        y_pred.append(pw)
        y_probs.append([a / 100, d / 100, h / 100])  # order: away, draw, home
        n += 1

    if n == 0:
        return {"n": 0}

    label_to_idx = {"away": 0, "draw": 1, "home": 2}
    yt = np.array([label_to_idx[y] for y in y_true])
    yp = np.array([label_to_idx[y] for y in y_pred])
    prob_arr = np.array(y_probs)
    prob_arr = np.clip(prob_arr, 0.01, 0.99)
    prob_arr = prob_arr / prob_arr.sum(axis=1, keepdims=True)

    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    try:
        ll = log_loss(yt, prob_arr, labels=[0, 1, 2])
    except Exception:
        ll = 0.0

    draw_pred = sum(1 for p in y_pred if p == "draw")
    draw_actual = sum(1 for y in y_true if y == "draw")
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "draw_predicted": draw_pred,
        "draw_actual": draw_actual,
    }


def _filter_recent_with_actual(predictions: list[dict], n_limit: int | None = None) -> list[dict]:
    """実績ありの予測を新しい順に取得"""
    wa = [p for p in predictions if (p.get("actual") or {}).get("winner")]
    wa.sort(key=lambda p: p.get("saved_at", ""), reverse=True)
    if n_limit:
        return wa[:n_limit]
    return wa


def run_monitoring_report() -> dict:
    """監視レポートを実行"""
    all_preds = load_all()

    # スキーマバージョン別カウント
    schema_counts = {}
    for p in all_preds:
        v = p.get("schema_version", "legacy")
        schema_counts[v] = schema_counts.get(v, 0) + 1

    # 実績ありの予測のみ抽出
    with_actual = _filter_recent_with_actual(all_preds)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_predictions": len(all_preds),
        "with_actual": len(with_actual),
        "schema_counts": schema_counts,
        "windows": {},
        "warnings": [],
        "auto_downgrade": {"should": False, "reason": ""},
    }

    # ローリングウィンドウごとに評価
    for window_name, window_size in ROLLING_WINDOWS.items():
        recent = _filter_recent_with_actual(all_preds, n_limit=window_size)
        if not recent:
            continue

        primary_m = _metrics(recent, "prediction")
        baseline_m = _metrics(recent, "baseline_prediction")
        shadow_m = _metrics(recent, "shadow_prediction")

        n = primary_m.get("n", 0)
        # baselineが存在する場合のみ比較
        b_n = baseline_m.get("n", 0)
        level, reasons = ("GREEN", ["baseline不在"]) if b_n == 0 else \
                         evaluate_warning_level(primary_m, baseline_m, min(n, b_n))

        report["windows"][window_name] = {
            "window_size": window_size,
            "primary": primary_m,
            "baseline": baseline_m,
            "shadow": shadow_m,
            "level": level,
            "reasons": reasons,
        }

    # 自動降格判定 (medium/formalウィンドウで決定)
    for ck in ["formal", "medium", "short"]:
        w = report["windows"].get(ck)
        if not w:
            continue
        level = w["level"]
        n = w["primary"].get("n", 0)
        should_down, reason = should_auto_downgrade(level, n)
        if should_down:
            report["auto_downgrade"] = {
                "should": True,
                "reason": reason,
                "window": ck,
                "level": level,
            }
            break

    # 警告メッセージ
    for wn, w in report["windows"].items():
        if w["level"] in ("YELLOW", "ORANGE", "RED"):
            report["warnings"].append({
                "window": wn,
                "level": w["level"],
                "reasons": w["reasons"],
            })

    return report


def print_report(report: dict) -> None:
    print("=" * 78)
    print("  Primary Model 監視レポート")
    print("=" * 78)
    print(f"生成日時: {report['generated_at']}")
    print(f"スキーマバージョン: {PREDICTION_SCHEMA_VERSION}")
    print(f"総予測数: {report['total_predictions']}")
    print(f"実績付き: {report['with_actual']}")
    print(f"スキーマ別: {report['schema_counts']}")
    print()

    for wn, w in report["windows"].items():
        print(f"--- {wn} ウィンドウ (最大{w['window_size']}試合) ---")
        for model_key, label in [("primary", "Primary (hybrid_v9.1)"),
                                  ("baseline", "Baseline (v7)"),
                                  ("shadow", "Shadow (v8.1)")]:
            m = w[model_key]
            if m.get("n", 0) == 0:
                print(f"  {label:<28} データなし")
                continue
            print(f"  {label:<28} n={m['n']:>3} "
                  f"acc={m.get('accuracy',0):.4f} F1={m.get('f1_macro',0):.4f} "
                  f"logL={m.get('log_loss',0):.4f} draw#={m.get('draw_predicted',0)}")
        print(f"  Level: {w['level']} - {WARNING_LEVELS.get(w['level'],'')}")
        for r in w["reasons"]:
            print(f"    - {r}")
        print()

    if report["warnings"]:
        print("=== 警告 ===")
        for w in report["warnings"]:
            print(f"  [{w['level']}] {w['window']}: {w['reasons']}")
        print()

    print("=== 自動降格判定 ===")
    ad = report["auto_downgrade"]
    if ad["should"]:
        print(f"  推奨: 降格 ({ad['window']} window, level={ad['level']})")
        print(f"  理由: {ad['reason']}")
        print(f"  対応: scripts/predict_logic.py の PRIMARY_MODEL_VERSION を 'v7_refined' に変更")
    else:
        print(f"  推奨: 現状維持 ({ad['reason']})")


if __name__ == "__main__":
    report = run_monitoring_report()
    print_report(report)

    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "monitoring_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")
