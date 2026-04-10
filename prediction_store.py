"""
prediction_store.py - 予測履歴の保存・管理
data/predictions.json に永続化
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

STORE_PATH = Path(__file__).parent / "data" / "predictions.json"

_WINNER_LABEL_TO_CODE = {"ホーム勝利": "home", "引き分け": "draw", "アウェー勝利": "away"}
_WINNER_CODE_TO_LABEL = {v: k for k, v in _WINNER_LABEL_TO_CODE.items()}


def load_all() -> list[dict]:
    """保存済み予測を全件読み込む（新しい順）"""
    if not STORE_PATH.exists():
        return []
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        preds = data.get("predictions", [])
        return sorted(preds, key=lambda p: p.get("saved_at", ""), reverse=True)
    except Exception:
        return []


def _write_all(predictions: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(
        json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _format_side_prediction(pred: dict, version: str) -> dict:
    """副次予測 (baseline/shadow) の保存用フォーマット"""
    if not pred:
        return None
    h = int(pred.get("home_win_prob", 40))
    d = int(pred.get("draw_prob", 25))
    a = int(pred.get("away_win_prob", 35))
    if h >= a and h >= d:
        w = "home"
    elif a > h and a >= d:
        w = "away"
    else:
        w = "draw"
    return {
        "home_win_prob": h,
        "draw_prob": d,
        "away_win_prob": a,
        "pred_winner": w,
        "model_version": pred.get("model_version", version),
    }


def save_prediction(
    division: str,
    match: dict,
    prediction: dict,
    shadow_prediction: dict | None = None,
    baseline_prediction: dict | None = None,
    model_version: str = "hybrid_v9.1",
    baseline_model_version: str = "v7_refined",
) -> str:
    """
    予測を保存してIDを返す。
    同じ日付+チーム組み合わせの予測は上書きする。

    Parameters
    ----------
    prediction : Primary modelの予測 (UI表示用)
    baseline_prediction : Baseline model (v7) の予測 (fallback/比較用)
    shadow_prediction : Shadow modelの予測 (v8.1等, 内部ログ用)
    model_version : Primary modelのバージョン識別子
    baseline_model_version : Baseline modelのバージョン識別子
    """
    predictions = load_all()
    key = f"{match['date']}_{match['home']}_{match['away']}"

    # 同一試合の既存予測を削除（上書き）
    predictions = [p for p in predictions if p.get("_key") != key]

    h = int(prediction.get("home_win_prob", 40))
    d = int(prediction.get("draw_prob",     25))
    a = int(prediction.get("away_win_prob", 35))
    if h >= a and h >= d:
        pred_winner = "home"
    elif a > h and a >= d:
        pred_winner = "away"
    else:
        pred_winner = "draw"

    shadow_entry = _format_side_prediction(shadow_prediction, "shadow")
    baseline_entry = _format_side_prediction(baseline_prediction, baseline_model_version)

    pred_id = str(uuid.uuid4())[:8]
    predictions.insert(0, {
        "id":       pred_id,
        "_key":     key,
        "saved_at": datetime.now().isoformat(),
        "division": division,
        "match":    match,
        "model_version": model_version,
        "baseline_model_version": baseline_model_version,
        "role":     "primary",
        "prediction": {
            "home_win_prob":   h,
            "draw_prob":       d,
            "away_win_prob":   a,
            "predicted_score": prediction.get("predicted_score", "?-?"),
            "confidence":      prediction.get("confidence", "medium"),
            "pred_winner":     pred_winner,
            "model":           prediction.get("model", ""),
            "hybrid_selection": prediction.get("hybrid_selection", ""),
        },
        "baseline_prediction": baseline_entry,
        "shadow_prediction":   shadow_entry,
        "actual": None,
    })
    _write_all(predictions)
    return pred_id


def update_actual(pred_id: str, score: str, winner_label: str) -> bool:
    """
    実際の結果を更新。
    winner_label: 'ホーム勝利' | '引き分け' | 'アウェー勝利'
    """
    predictions = load_all()
    for p in predictions:
        if p["id"] == pred_id:
            p["actual"] = {
                "score":        score.strip(),
                "winner":       _WINNER_LABEL_TO_CODE.get(winner_label, ""),
                "winner_label": winner_label,
                "recorded_at":  datetime.now().isoformat(),
            }
            _write_all(predictions)
            return True
    return False


def delete_prediction(pred_id: str) -> bool:
    """予測を削除"""
    predictions = load_all()
    new = [p for p in predictions if p["id"] != pred_id]
    if len(new) == len(predictions):
        return False
    _write_all(new)
    return True


def get_accuracy_stats(predictions: list[dict]) -> dict:
    """正答率統計を計算"""
    total = len(predictions)
    with_actual = [p for p in predictions if p.get("actual") and p["actual"].get("winner")]
    correct = [
        p for p in with_actual
        if p["actual"]["winner"] == p["prediction"].get("pred_winner")
    ]
    by_conf: dict[str, dict] = {}
    for p in with_actual:
        conf = p["prediction"].get("confidence", "medium")
        is_correct = p["actual"]["winner"] == p["prediction"].get("pred_winner")
        if conf not in by_conf:
            by_conf[conf] = {"total": 0, "correct": 0}
        by_conf[conf]["total"] += 1
        if is_correct:
            by_conf[conf]["correct"] += 1

    return {
        "total":       total,
        "with_actual": len(with_actual),
        "correct":     len(correct),
        "accuracy":    len(correct) / len(with_actual) if with_actual else None,
        "by_conf":     by_conf,
    }
