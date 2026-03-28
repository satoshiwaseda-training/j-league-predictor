"""
predictor.py - Gemini 2.5 Flash API を使ったJリーグ試合予測エンジン
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Gemini クライアント初期化
# ─────────────────────────────────────────────

def _get_gemini_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            import streamlit as _st
            for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                try:
                    api_key = str(_st.secrets[_k])
                    if api_key:
                        os.environ["GEMINI_API_KEY"] = api_key
                        break
                except Exception:
                    pass
        except Exception:
            pass
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY が設定されていません。.env を確認してください。")
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────
# メイン予測関数
# ─────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    home_stats: dict[str, Any],
    away_stats: dict[str, Any],
    weather: dict[str, Any],
    h2h: dict[str, Any],
    home_form: list[str],
    away_form: list[str],
    home_injuries: list[dict],
    away_injuries: list[dict],
) -> dict[str, Any]:
    """
    Gemini 2.5 Flash で試合結果を予測。

    Returns
    -------
    {
      "home_win_prob": int,   # ホーム勝利確率 (%)
      "draw_prob": int,       # 引き分け確率 (%)
      "away_win_prob": int,   # アウェー勝利確率 (%)
      "predicted_score": str, # 予想スコア "X-Y"
      "confidence": str,      # "high" / "medium" / "low"
      "reasoning": str,       # 日本語の科学的根拠
      "key_factors": list[str],
      "model": str,
    }
    """
    prompt = _build_prompt(
        home_team, away_team,
        home_stats, away_stats,
        weather, h2h,
        home_form, away_form,
        home_injuries, away_injuries,
    )

    try:
        client = _get_gemini_client()
        from google import genai as _genai

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        raw = response.text.strip()
        # JSON ブロックを抽出
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        result = json.loads(raw)
        result["model"] = "gemini-2.5-flash"
        _normalize_probs(result)
        return result

    except ValueError as exc:
        # APIキー未設定
        logger.error("Gemini init error: %s", exc)
        return {"error": str(exc), "model": "none"}

    except Exception as exc:
        logger.error("Gemini prediction error: %s", exc)
        return _fallback_prediction(home_team, away_team, home_stats, away_stats)


# ─────────────────────────────────────────────
# プロンプト生成
# ─────────────────────────────────────────────

def _build_prompt(
    home: str, away: str,
    hs: dict, as_: dict,
    weather: dict, h2h: dict,
    home_form: list, away_form: list,
    home_inj: list, away_inj: list,
) -> str:
    def form_str(f: list) -> str:
        return " ".join(f) if f else "不明"

    def inj_str(lst: list) -> str:
        if not lst:
            return "なし"
        return "、".join(f"{i.get('player','不明')}({i.get('status','欠場')})" for i in lst[:3])

    weather_desc = (
        f"{weather.get('description','不明')} "
        f"気温{weather.get('temp_avg','?')}°C "
        f"降水{weather.get('precipitation','?')}mm "
        f"風速{weather.get('wind_speed','?')}km/h "
        f"疲労影響スコア: {weather.get('fatigue_factor', 0):.2f}"
    )

    h2h_str = (
        f"過去{h2h.get('total',0)}試合: "
        f"{home}{h2h.get('home_wins',0)}勝 "
        f"引分{h2h.get('draws',0)} "
        f"{away}{h2h.get('away_wins',0)}勝"
    )

    prompt = f"""あなたはJリーグの試合結果を科学的に予測するAIアナリストです。
以下のデータを分析し、JSON形式で予測を返してください。

## 試合情報
- ホーム: {home}
- アウェー: {away}

## ホームチーム ({home}) データ
- リーグ順位: {hs.get('順位', '?')}位
- 勝点: {hs.get('勝点', '?')}
- 成績: {hs.get('試合','?')}試合 {hs.get('勝','?')}勝{hs.get('分','?')}分{hs.get('負','?')}敗
- 得点/失点: {hs.get('得点','?')}/{hs.get('失点','?')} (得失点差: {hs.get('得失点差','?')})
- 直近5試合: {form_str(home_form)}
- 怪我/停止: {inj_str(home_inj)}

## アウェーチーム ({away}) データ
- リーグ順位: {as_.get('順位', '?')}位
- 勝点: {as_.get('勝点', '?')}
- 成績: {as_.get('試合','?')}試合 {as_.get('勝','?')}勝{as_.get('分','?')}分{as_.get('負','?')}敗
- 得点/失点: {as_.get('得点','?')}/{as_.get('失点','?')} (得失点差: {as_.get('得失点差','?')})
- 直近5試合: {form_str(away_form)}
- 怪我/停止: {inj_str(away_inj)}

## 対戦成績 (H2H)
{h2h_str}

## 試合会場の天気
{weather_desc}

## 分析・予測の依頼
以下の科学的要素を必ず考慮して予測してください:
1. ELO レーティング的な強度差
2. 直近フォーム（モメンタム）
3. ホームアドバンテージ（平均 +0.4 ゴール効果）
4. H2H 心理的優位性
5. 天気・気温による疲労影響
6. 怪我・出場停止による戦力低下

## 出力形式（必ず以下の JSON のみを返すこと）
{{
  "home_win_prob": <ホーム勝利確率 0-100 整数>,
  "draw_prob": <引き分け確率 0-100 整数>,
  "away_win_prob": <アウェー勝利確率 0-100 整数>,
  "predicted_score": "<ホーム得点>-<アウェー得点>",
  "confidence": "<high|medium|low>",
  "reasoning": "<300字以上の日本語による科学的根拠。各要因がどう確率に影響したか具体的に>",
  "key_factors": [
    "<影響要因1>",
    "<影響要因2>",
    "<影響要因3>"
  ]
}}

注意: home_win_prob + draw_prob + away_win_prob = 100 にすること。
"""
    return prompt


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def _normalize_probs(result: dict) -> None:
    """確率の合計を 100 に正規化"""
    h = int(result.get("home_win_prob", 33))
    d = int(result.get("draw_prob", 33))
    a = int(result.get("away_win_prob", 34))
    total = h + d + a
    if total != 100 and total > 0:
        result["home_win_prob"] = round(h / total * 100)
        result["draw_prob"] = round(d / total * 100)
        result["away_win_prob"] = 100 - result["home_win_prob"] - result["draw_prob"]


def _fallback_prediction(
    home: str, away: str,
    hs: dict, as_: dict,
) -> dict:
    """API エラー時の統計ベースフォールバック予測"""
    h_pts = int(hs.get("勝点", 30))
    a_pts = int(as_.get("勝点", 30))
    total = h_pts + a_pts + 10  # ホームアドバンテージ補正

    # ホームアドバンテージ: +5pt 相当
    h_prob = round((h_pts + 5) / total * 100)
    a_prob = round(a_pts / total * 100)
    d_prob = 100 - h_prob - a_prob

    return {
        "home_win_prob": max(h_prob, 10),
        "draw_prob": max(d_prob, 10),
        "away_win_prob": max(a_prob, 10),
        "predicted_score": "1-1",
        "confidence": "low",
        "reasoning": (
            "⚠️ Gemini API に接続できなかったため、統計ベースのフォールバック予測です。"
            "勝点差とホームアドバンテージ（+5pt）のみで計算しています。"
            ".env の GEMINI_API_KEY を設定すると AI 予測が有効になります。"
        ),
        "key_factors": [
            f"ホームチーム勝点: {h_pts}",
            f"アウェーチーム勝点: {a_pts}",
            "ホームアドバンテージ補正 (+5pt)",
        ],
        "model": "fallback-statistical",
    }
