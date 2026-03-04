"""
scripts/feedback_loop.py - 予測フィードバックループ

過去の予測と実結果を照合し、Gemini 2.0 Flash による
敗因分析・パラメータ調整提案・新指標提案を生成する。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. 予測精度分析
# ─────────────────────────────────────────────

def analyze_predictions(predictions: list[dict]) -> dict:
    """
    予測履歴を分析して正答率・誤差・傾向を返す。

    Returns:
        total, with_actual, correct, accuracy,
        wrong: [予測ミス試合リスト],
        by_outcome: {home/draw/away: {predicted, correct, accuracy}},
        by_confidence: {high/medium/low: {total, correct, accuracy}},
        upset_rate: ジャイキリ（下位チームが勝利）の発生率,
    """
    with_actual = [
        p for p in predictions
        if p.get("actual") and p["actual"].get("winner")
    ]

    if not with_actual:
        return {
            "total": len(predictions), "with_actual": 0,
            "correct": 0, "accuracy": None,
            "wrong": [], "by_outcome": {}, "by_confidence": {},
            "upset_rate": None,
        }

    correct, wrong = [], []
    for p in with_actual:
        if p["prediction"].get("pred_winner") == p["actual"]["winner"]:
            correct.append(p)
        else:
            wrong.append(p)

    # 予測カテゴリ別
    by_outcome: dict[str, dict] = {}
    for outcome in ("home", "draw", "away"):
        predicted = [p for p in with_actual if p["prediction"].get("pred_winner") == outcome]
        hit = [p for p in predicted if p["actual"]["winner"] == outcome]
        by_outcome[outcome] = {
            "predicted": len(predicted),
            "correct":   len(hit),
            "accuracy":  len(hit) / len(predicted) if predicted else None,
        }

    # 信頼度別
    by_conf: dict[str, dict] = {}
    for conf in ("high", "medium", "low"):
        cp = [p for p in with_actual if p["prediction"].get("confidence") == conf]
        ch = [p for p in cp if p["actual"]["winner"] == p["prediction"].get("pred_winner")]
        by_conf[conf] = {
            "total":    len(cp),
            "correct":  len(ch),
            "accuracy": len(ch) / len(cp) if cp else None,
        }

    # ジャイアントキリング発生率（アウェー勝利のうち予測がホーム勝利だったもの）
    upset_cases = [
        p for p in with_actual
        if p["actual"]["winner"] == "away"
        and p["prediction"].get("pred_winner") == "home"
        and int(p["prediction"].get("home_win_prob", 0)) >= 50
    ]
    upset_rate = len(upset_cases) / len(with_actual) if with_actual else None

    return {
        "total":         len(predictions),
        "with_actual":   len(with_actual),
        "correct":       len(correct),
        "accuracy":      len(correct) / len(with_actual),
        "wrong":         wrong,
        "by_outcome":    by_outcome,
        "by_confidence": by_conf,
        "upset_rate":    upset_rate,
        "upset_cases":   upset_cases,
    }


# ─────────────────────────────────────────────
# 2. Gemini による敗因分析 & 改善提案
# ─────────────────────────────────────────────

def _resolve_api_key() -> str:
    ak = os.getenv("GEMINI_API_KEY", "")
    if not ak:
        try:
            import streamlit as _st
            for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                try:
                    ak = str(_st.secrets[_k])
                    if ak:
                        os.environ["GEMINI_API_KEY"] = ak
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return ak


def ask_gemini_for_analysis(
    wrong_preds: list[dict],
    all_preds:   list[dict],
    weights:     dict[str, float],
) -> dict:
    """
    Gemini 2.0 Flash に不正解予測の敗因分析と改善提案を依頼する。

    Returns dict with keys:
        defeat_causes: list[str]
        weight_adjustments: list[{param, current, suggested, reason}]
        new_indicators: list[{name, description, expected_impact}]
        summary: str
        error: str | None
    """
    api_key = _resolve_api_key()
    if not api_key:
        return {
            "error": "GEMINI_API_KEY 未設定",
            "defeat_causes": [], "weight_adjustments": [],
            "new_indicators": [], "summary": "APIキー未設定のため分析不可",
        }

    # 全体統計
    total_wa = len([p for p in all_preds if p.get("actual") and p["actual"].get("winner")])
    accuracy = (total_wa - len(wrong_preds)) / total_wa if total_wa else None
    acc_str = f"{accuracy:.1%}" if accuracy is not None else "不明"

    # 不正解試合の詳細（最大12試合）
    cases_lines: list[str] = []
    for p in wrong_preds[:12]:
        m    = p["match"]
        pred = p["prediction"]
        act  = p.get("actual", {})
        h_prob = pred.get("home_win_prob", "?")
        d_prob = pred.get("draw_prob",     "?")
        a_prob = pred.get("away_win_prob", "?")
        cases_lines.append(
            f"  - {m['date']} 【{m.get('division','J1')}】"
            f"{m['home']} vs {m['away']}\n"
            f"    予測: {pred.get('pred_winner','?')} "
            f"(H{h_prob}% / D{d_prob}% / A{a_prob}%) 信頼度={pred.get('confidence','?')}\n"
            f"    実結果: {act.get('winner','?')} スコア={act.get('score','?-?')}"
        )
    cases_text = "\n".join(cases_lines) if cases_lines else "  （不正解データなし）"

    # 傾向サマリー
    home_misses  = sum(1 for p in wrong_preds if p["prediction"].get("pred_winner") == "home")
    draw_misses  = sum(1 for p in wrong_preds if p["prediction"].get("pred_winner") == "draw")
    away_misses  = sum(1 for p in wrong_preds if p["prediction"].get("pred_winner") == "away")
    high_misses  = sum(1 for p in wrong_preds if p["prediction"].get("confidence") == "high")

    weights_text = "\n".join(
        f"  {k}: {v:.2f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])
    )

    prompt = f"""あなたはJリーグの戦力分析・機械学習モデル改善エキスパートです。
以下は私のJリーグ勝敗予測モデルが外れた試合の詳細です。

## 全体精度
- 結果入力済み試合: {total_wa}試合
- 現在の正答率: {acc_str}
- 不正解: {len(wrong_preds)}試合

## 不正解の傾向
- ホーム勝利と予測してハズレ: {home_misses}試合
- ドローと予測してハズレ: {draw_misses}試合
- アウェー勝利と予測してハズレ: {away_misses}試合
- 「高信頼度」でハズレ: {high_misses}試合

## 不正解試合の詳細
{cases_text}

## 現在のモデル重み（合計1.00）
{weights_text}

## 分析タスク
1. **敗因分析**: これらの不正解試合の共通パターンと見落とされた要因を分析してください（3点）。
2. **重み微調整提案**: 現在の重みのうち増減すべきものを最大4つ提案してください。
   - 変更幅は ±0.01〜±0.04 以内
   - 全パラメータの合計が1.00を維持するよう調整
3. **新指標提案**: モデルにまだない指標で追加価値があるものを1〜2個提案してください。

必ず以下のJSON形式のみで回答してください（説明文は不要）:
{{
  "defeat_causes": ["原因1の日本語説明", "原因2", "原因3"],
  "weight_adjustments": [
    {{"param": "パラメータ名", "current": 0.00, "suggested": 0.00, "reason": "理由（日本語）"}},
    {{"param": "...", "current": 0.00, "suggested": 0.00, "reason": "..."}}
  ],
  "new_indicators": [
    {{"name": "指標名", "description": "説明（日本語）", "expected_impact": "期待効果"}},
    {{"name": "...", "description": "...", "expected_impact": "..."}}
  ],
  "summary": "全体的な改善方針を1〜2文で（日本語）"
}}"""

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.35,
            },
        )
        result = json.loads(response.text)
        result["error"] = None
        result["generated_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error("Gemini feedback analysis failed: %s", e)
        return {
            "error":              str(e),
            "defeat_causes":      [],
            "weight_adjustments": [],
            "new_indicators":     [],
            "summary":            f"分析失敗: {e}",
            "generated_at":       datetime.now().isoformat(),
        }


# ─────────────────────────────────────────────
# 3. シーズンバックテスト用ユーティリティ
# ─────────────────────────────────────────────

def sync_results_to_store(division: str = "j1") -> tuple[int, int]:
    """
    get_past_results() を使って prediction_store の未入力予測に
    実結果を自動入力する。
    Returns (synced_count, skipped_count)
    """
    from data_fetcher import get_past_results
    from prediction_store import load_all, update_actual

    preds  = load_all()
    past   = get_past_results(division)
    result_map = {
        (r["home"], r["away"], r["date"]): r for r in past
    }

    synced = skipped = 0
    for p in preds:
        if p.get("actual") and p["actual"].get("winner"):
            skipped += 1
            continue
        m   = p["match"]
        key = (m["home"], m["away"], m["date"])
        if key in result_map:
            r = result_map[key]
            winner_map = {"home": "ホーム勝利", "draw": "引き分け", "away": "アウェー勝利"}
            label = winner_map.get(r["winner"], "")
            if label:
                update_actual(p["id"], r["score"], label)
                synced += 1

    return synced, skipped
