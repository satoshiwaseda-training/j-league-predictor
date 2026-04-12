"""
weekend_review.py - 予測 vs 実結果の分析・週末レビュー生成

責務:
  1. 予測と実結果の突合 (evaluate_weekend_predictions)
  2. 週末レビュー表の構築 (build_weekend_review_table)
  3. 要約生成 (summarize_weekend_review)
  4. 分析: 高確信外し / ドロー取り逃し / 波乱検知 / モデルの癖 / 品質影響

データリーク防止:
  - 予測履歴がなければ後付け推定はしない (参考値として別扱い)
  - 予測時点の値と実結果を厳密に突合
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
REVIEW_CSV_PATH = DATA_DIR / "weekend_review.csv"
SUMMARY_JSON_PATH = DATA_DIR / "weekend_summary.json"


# ─── ヘルパー ─────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _brier_component(prob_h: float, prob_d: float, prob_a: float, actual: str) -> float:
    """単一試合の Brier Score 寄与 (低いほど良い)"""
    p = [prob_h / 100.0, prob_d / 100.0, prob_a / 100.0]
    actual_vec = [0.0, 0.0, 0.0]
    if actual == "H":
        actual_vec[0] = 1.0
    elif actual == "D":
        actual_vec[1] = 1.0
    elif actual == "A":
        actual_vec[2] = 1.0
    return sum((p[i] - actual_vec[i]) ** 2 for i in range(3))


def _logloss_component(prob_h: float, prob_d: float, prob_a: float, actual: str) -> float:
    """単一試合の LogLoss 寄与 (低いほど良い)"""
    eps = 1e-10
    probs = {"H": max(prob_h / 100.0, eps), "D": max(prob_d / 100.0, eps), "A": max(prob_a / 100.0, eps)}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}  # 正規化
    if actual in probs:
        return -math.log(probs[actual])
    return -math.log(eps)


# ─── 1. 予測と実結果の突合 ──────────────────────────────

def evaluate_weekend_predictions(
    results_df: pd.DataFrame,
    predictions: list[dict] | None = None,
) -> list[dict]:
    """
    今週末の結果と予測履歴を突合し、各試合の評価を返す。

    Parameters
    ----------
    results_df : fetch_weekend_results() の出力
    predictions : prediction_store.load_all() の出力 (None なら自動読み込み)

    Returns
    -------
    list[dict] - 各試合の評価レコード
    """
    if predictions is None:
        from prediction_store import load_all
        predictions = load_all()

    if results_df.empty:
        return []

    # 予測を _key でインデックス
    pred_by_key: dict[str, dict] = {}
    for p in predictions:
        key = p.get("_key", "")
        if key:
            pred_by_key[key] = p

    evaluations: list[dict] = []

    for _, row in results_df.iterrows():
        date_str = str(row.get("date", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        result_code = str(row.get("result", ""))
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        competition = str(row.get("competition", ""))

        pkey = f"{date_str}_{home}_{away}"
        pred = pred_by_key.get(pkey)

        has_prediction = pred is not None
        pred_data = pred.get("prediction", {}) if pred else {}

        prob_h = _safe_float(pred_data.get("home_win_prob"), 33.3)
        prob_d = _safe_float(pred_data.get("draw_prob"), 33.3)
        prob_a = _safe_float(pred_data.get("away_win_prob"), 33.3)
        pred_winner = pred_data.get("pred_winner", "")
        confidence = pred_data.get("confidence", "")
        pred_score = pred_data.get("predicted_score", "")
        model_version = pred.get("model_version", "") if pred else ""

        # 実結果コード → pred_winner 互換形式
        actual_winner = ""
        if result_code == "H":
            actual_winner = "home"
        elif result_code == "D":
            actual_winner = "draw"
        elif result_code == "A":
            actual_winner = "away"

        # 的中判定
        is_correct = pred_winner == actual_winner if has_prediction else None

        # Brier / LogLoss
        brier = _brier_component(prob_h, prob_d, prob_a, result_code) if has_prediction else None
        logloss = _logloss_component(prob_h, prob_d, prob_a, result_code) if has_prediction else None

        # max_prob
        max_prob = max(prob_h, prob_d, prob_a) if has_prediction else None

        # 予測ラベル
        pred_label = ""
        if has_prediction:
            if confidence == "high":
                pred_label = "高確信"
            elif confidence == "medium":
                pred_label = "中確信"
            elif confidence == "low":
                pred_label = "低確信"
            else:
                pred_label = "不明"

        # Gemini コメントの有無
        has_gemini = bool(pred_data.get("model", "")) and "gemini" in str(pred_data.get("model", "")).lower() if pred else False

        # 波乱判定: 予測上位でない結果が出た場合
        is_upset = False
        if has_prediction and not is_correct and max_prob and max_prob >= 50:
            is_upset = True

        # ドロー注意判定: 実結果Dで予測draw_probが低かった
        draw_miss = False
        if has_prediction and result_code == "D" and prob_d < 28:
            draw_miss = True

        # データ品質ランク
        quality_rank = _infer_quality_rank(pred) if pred else "D"

        # ── fan/travel 補正監視データ ──
        adj = pred.get("adjustments") if pred else None
        fan_applied = bool(adj and adj.get("fan_applied")) if adj else False
        travel_applied = bool(adj and adj.get("travel_applied")) if adj else False
        fan_value = adj.get("fan_value", 0.0) if adj else None
        travel_value = adj.get("travel_value", 0.0) if adj else None
        argmax_changed = bool(adj and adj.get("argmax_changed")) if adj else False
        pre_probs = None
        post_probs = None
        if adj:
            pre_probs = f"{adj.get('pre_h','?')}-{adj.get('pre_d','?')}-{adj.get('pre_a','?')}"
            post_probs = f"{adj.get('post_h','?')}-{adj.get('post_d','?')}-{adj.get('post_a','?')}"

        evaluations.append({
            "date": date_str,
            "competition": competition,
            "home_team": home,
            "away_team": away,
            "match_card": f"{home} vs {away}",
            "home_score": home_score,
            "away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_result": result_code,
            "has_prediction": has_prediction,
            "pred_prob_h": prob_h if has_prediction else None,
            "pred_prob_d": prob_d if has_prediction else None,
            "pred_prob_a": prob_a if has_prediction else None,
            "pred_winner": pred_winner,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "confidence": confidence,
            "max_prob": max_prob,
            "is_correct": is_correct,
            "brier": round(brier, 4) if brier is not None else None,
            "logloss": round(logloss, 4) if logloss is not None else None,
            "has_gemini": has_gemini,
            "is_upset": is_upset,
            "draw_miss": draw_miss,
            "quality_rank": quality_rank,
            "model_version": model_version,
            "fan_applied": fan_applied,
            "travel_applied": travel_applied,
            "fan_value": fan_value,
            "travel_value": travel_value,
            "pre_adjust_probs": pre_probs,
            "post_adjust_probs": post_probs,
            "argmax_changed": argmax_changed,
        })

    return evaluations


def _infer_quality_rank(pred: dict) -> str:
    """予測レコードからデータ品質ランクを推定"""
    if not pred:
        return "D"
    pred_data = pred.get("prediction", {})
    model = str(pred_data.get("model", ""))
    has_gemini = "gemini" in model.lower()

    # shadow/baseline の有無も品質の指標
    has_shadow = pred.get("shadow_prediction") is not None
    has_baseline = pred.get("baseline_prediction") is not None

    if has_gemini and has_shadow:
        return "A"
    elif has_gemini or has_shadow:
        return "B"
    elif has_baseline:
        return "C"
    return "C"


# ─── 2. 週末レビュー表の構築 ─────────────────────────────

def build_weekend_review_table(evaluations: list[dict]) -> pd.DataFrame:
    """評価リストを表示用 DataFrame に変換"""
    if not evaluations:
        return pd.DataFrame()

    df = pd.DataFrame(evaluations)

    # 表示用カラム整理
    display_cols = [
        "date", "competition", "match_card", "actual_score", "actual_result",
        "has_prediction", "pred_prob_h", "pred_prob_d", "pred_prob_a",
        "pred_label", "is_correct", "brier", "logloss",
        "has_gemini", "is_upset", "draw_miss", "quality_rank",
        "fan_applied", "travel_applied", "argmax_changed",
    ]
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].copy()

    # 日本語カラム名
    rename_map = {
        "date": "日付",
        "competition": "リーグ",
        "match_card": "試合",
        "actual_score": "スコア",
        "actual_result": "結果",
        "has_prediction": "予測あり",
        "pred_prob_h": "H%",
        "pred_prob_d": "D%",
        "pred_prob_a": "A%",
        "pred_label": "確信度",
        "is_correct": "的中",
        "brier": "Brier",
        "logloss": "LogLoss",
        "has_gemini": "Gemini",
        "is_upset": "波乱",
        "draw_miss": "D取逃",
        "quality_rank": "品質",
        "fan_applied": "Fan補正",
        "travel_applied": "Travel発火",
        "argmax_changed": "ラベル変化",
    }
    df_display = df_display.rename(columns=rename_map)

    # CSV 保存
    df.to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")

    return df_display


# ─── 3. 要約生成 ──────────────────────────────────────────

def summarize_weekend_review(evaluations: list[dict]) -> dict:
    """
    週末レビューの要約を生成。

    Returns
    -------
    dict with summary statistics and analysis
    """
    if not evaluations:
        return {"total": 0, "message": "対象試合なし"}

    total = len(evaluations)
    with_pred = [e for e in evaluations if e.get("has_prediction")]
    without_pred = total - len(with_pred)

    # 的中率
    correct_list = [e for e in with_pred if e.get("is_correct") is True]
    incorrect_list = [e for e in with_pred if e.get("is_correct") is False]
    accuracy = len(correct_list) / len(with_pred) if with_pred else None

    # Brier / LogLoss
    brier_vals = [e["brier"] for e in with_pred if e.get("brier") is not None]
    logloss_vals = [e["logloss"] for e in with_pred if e.get("logloss") is not None]
    avg_brier = sum(brier_vals) / len(brier_vals) if brier_vals else None
    avg_logloss = sum(logloss_vals) / len(logloss_vals) if logloss_vals else None

    # ドロー分析
    actual_draws = [e for e in evaluations if e.get("actual_result") == "D"]
    draw_correct = [e for e in actual_draws if e.get("has_prediction") and e.get("is_correct")]
    draw_misses = [e for e in with_pred if e.get("draw_miss")]

    # 波乱
    upsets = [e for e in with_pred if e.get("is_upset")]

    # 高確信外し (max_prob >= 60 なのに外れ)
    high_conf_misses = [
        e for e in with_pred
        if not e.get("is_correct") and _safe_float(e.get("max_prob")) >= 60
    ]

    # 低確信的中 (max_prob < 45 で的中)
    low_conf_hits = [
        e for e in with_pred
        if e.get("is_correct") and _safe_float(e.get("max_prob")) < 45
    ]

    # 品質ランク別成績
    quality_stats: dict[str, dict] = {}
    for e in with_pred:
        qr = e.get("quality_rank", "?")
        if qr not in quality_stats:
            quality_stats[qr] = {"total": 0, "correct": 0}
        quality_stats[qr]["total"] += 1
        if e.get("is_correct"):
            quality_stats[qr]["correct"] += 1
    for qr in quality_stats:
        t = quality_stats[qr]["total"]
        c = quality_stats[qr]["correct"]
        quality_stats[qr]["accuracy"] = round(c / t, 3) if t > 0 else None

    # Gemini 補正分析
    gemini_matches = [e for e in with_pred if e.get("has_gemini")]
    non_gemini = [e for e in with_pred if not e.get("has_gemini")]
    gemini_acc = (
        sum(1 for e in gemini_matches if e.get("is_correct")) / len(gemini_matches)
        if gemini_matches else None
    )
    non_gemini_acc = (
        sum(1 for e in non_gemini if e.get("is_correct")) / len(non_gemini)
        if non_gemini else None
    )

    # モデルの癖分析
    bias_analysis = _analyze_model_bias(with_pred)

    # 最も大きく外した試合
    worst_miss = None
    if incorrect_list:
        worst = max(incorrect_list, key=lambda e: _safe_float(e.get("brier"), 0))
        worst_miss = {
            "match": worst.get("match_card", ""),
            "score": worst.get("actual_score", ""),
            "brier": worst.get("brier"),
            "max_prob": worst.get("max_prob"),
        }

    # 最も評価しやすかった試合 (的中 + 最低Brier)
    best_hit = None
    if correct_list:
        best = min(correct_list, key=lambda e: _safe_float(e.get("brier"), 999))
        best_hit = {
            "match": best.get("match_card", ""),
            "score": best.get("actual_score", ""),
            "brier": best.get("brier"),
            "max_prob": best.get("max_prob"),
        }

    # ── fan/travel 補正効果分析 ──
    fan_applied = [e for e in with_pred if e.get("fan_applied")]
    travel_applied = [e for e in with_pred if e.get("travel_applied")]
    argmax_changed = [e for e in with_pred if e.get("argmax_changed")]

    def _sub_metrics(subset):
        if not subset:
            return {"n": 0, "accuracy": None, "avg_brier": None, "avg_logloss": None}
        n = len(subset)
        c = sum(1 for e in subset if e.get("is_correct"))
        bs = [e["brier"] for e in subset if e.get("brier") is not None]
        ls = [e["logloss"] for e in subset if e.get("logloss") is not None]
        return {
            "n": n,
            "accuracy": round(c / n, 4) if n else None,
            "avg_brier": round(sum(bs) / len(bs), 4) if bs else None,
            "avg_logloss": round(sum(ls) / len(ls), 4) if ls else None,
        }

    fan_no_applied = [e for e in with_pred if not e.get("fan_applied")]
    adjustment_stats = {
        "fan_applied_count": len(fan_applied),
        "travel_applied_count": len(travel_applied),
        "argmax_changed_count": len(argmax_changed),
        "argmax_changed_matches": [
            {
                "match": e.get("match_card", ""),
                "pre": e.get("pre_adjust_probs", ""),
                "post": e.get("post_adjust_probs", ""),
                "actual": e.get("actual_result", ""),
                "correct": e.get("is_correct"),
            }
            for e in argmax_changed
        ],
        "fan_applied_metrics": _sub_metrics(fan_applied),
        "fan_not_applied_metrics": _sub_metrics(fan_no_applied),
        "travel_applied_metrics": _sub_metrics(travel_applied),
    }

    summary = {
        "total": total,
        "with_prediction": len(with_pred),
        "without_prediction": without_pred,
        "correct": len(correct_list),
        "incorrect": len(incorrect_list),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
        "avg_brier": round(avg_brier, 4) if avg_brier is not None else None,
        "avg_logloss": round(avg_logloss, 4) if avg_logloss is not None else None,
        "draw_total": len(actual_draws),
        "draw_correct": len(draw_correct),
        "draw_misses": len(draw_misses),
        "upset_count": len(upsets),
        "upsets": [
            {"match": u.get("match_card"), "score": u.get("actual_score"), "max_prob": u.get("max_prob")}
            for u in upsets
        ],
        "high_conf_misses": [
            {"match": m.get("match_card"), "score": m.get("actual_score"),
             "max_prob": m.get("max_prob"), "pred_winner": m.get("pred_winner")}
            for m in high_conf_misses
        ],
        "low_conf_hits": [
            {"match": h.get("match_card"), "score": h.get("actual_score"), "max_prob": h.get("max_prob")}
            for h in low_conf_hits
        ],
        "quality_stats": quality_stats,
        "gemini_accuracy": round(gemini_acc, 3) if gemini_acc is not None else None,
        "non_gemini_accuracy": round(non_gemini_acc, 3) if non_gemini_acc is not None else None,
        "bias_analysis": bias_analysis,
        "worst_miss": worst_miss,
        "best_hit": best_hit,
        "adjustment_stats": adjustment_stats,
    }

    # JSON 保存
    try:
        DATA_DIR.mkdir(exist_ok=True)
        SUMMARY_JSON_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("サマリー保存失敗: %s", e)

    return summary


def _analyze_model_bias(with_pred: list[dict]) -> dict:
    """モデルの癖を分析"""
    if not with_pred:
        return {}

    # ホーム過大評価: ホーム予測 > 実際のホーム勝率
    home_pred_count = sum(1 for e in with_pred if e.get("pred_winner") == "home")
    home_actual_count = sum(1 for e in with_pred if e.get("actual_result") == "H")
    home_total = len(with_pred)

    # アウェイ過小評価
    away_pred_count = sum(1 for e in with_pred if e.get("pred_winner") == "away")
    away_actual_count = sum(1 for e in with_pred if e.get("actual_result") == "A")

    # ドロー抑制
    draw_pred_count = sum(1 for e in with_pred if e.get("pred_winner") == "draw")
    draw_actual_count = sum(1 for e in with_pred if e.get("actual_result") == "D")

    # 平均予測確率
    avg_prob_h = sum(_safe_float(e.get("pred_prob_h")) for e in with_pred) / home_total
    avg_prob_d = sum(_safe_float(e.get("pred_prob_d")) for e in with_pred) / home_total
    avg_prob_a = sum(_safe_float(e.get("pred_prob_a")) for e in with_pred) / home_total

    # 実際の比率
    actual_h_rate = home_actual_count / home_total if home_total else 0
    actual_d_rate = draw_actual_count / home_total if home_total else 0
    actual_a_rate = away_actual_count / home_total if home_total else 0

    bias = {
        "home_pred_rate": round(home_pred_count / home_total, 3) if home_total else None,
        "home_actual_rate": round(actual_h_rate, 3),
        "home_overestimate": round(avg_prob_h / 100 - actual_h_rate, 3),
        "away_pred_rate": round(away_pred_count / home_total, 3) if home_total else None,
        "away_actual_rate": round(actual_a_rate, 3),
        "away_underestimate": round(actual_a_rate - avg_prob_a / 100, 3),
        "draw_pred_rate": round(draw_pred_count / home_total, 3) if home_total else None,
        "draw_actual_rate": round(actual_d_rate, 3),
        "draw_suppression": round(actual_d_rate - avg_prob_d / 100, 3),
        "avg_prob_h": round(avg_prob_h, 1),
        "avg_prob_d": round(avg_prob_d, 1),
        "avg_prob_a": round(avg_prob_a, 1),
    }

    # 警告生成
    warnings = []
    if bias["home_overestimate"] and bias["home_overestimate"] > 0.10:
        warnings.append(f"ホーム過大評価: 予測平均{avg_prob_h:.1f}% vs 実際{actual_h_rate*100:.1f}%")
    if bias["away_underestimate"] and bias["away_underestimate"] > 0.10:
        warnings.append(f"アウェイ過小評価: 予測平均{avg_prob_a:.1f}% vs 実際{actual_a_rate*100:.1f}%")
    if bias["draw_suppression"] and bias["draw_suppression"] > 0.08:
        warnings.append(f"ドロー抑制: 予測平均{avg_prob_d:.1f}% vs 実際{actual_d_rate*100:.1f}%")
    bias["warnings"] = warnings

    return bias


# ─── 統合レビュー関数 ─────────────────────────────────────

def run_weekend_review(
    results_df: pd.DataFrame,
    predictions: list[dict] | None = None,
) -> dict:
    """
    週末レビューを一括生成。

    Returns
    -------
    {
        "evaluations": list[dict],
        "review_table": DataFrame,
        "summary": dict,
    }
    """
    evaluations = evaluate_weekend_predictions(results_df, predictions)
    review_table = build_weekend_review_table(evaluations)
    summary = summarize_weekend_review(evaluations)

    return {
        "evaluations": evaluations,
        "review_table": review_table,
        "summary": summary,
    }
