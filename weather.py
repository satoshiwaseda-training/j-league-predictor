"""
weather.py - Open-Meteo 無料 API で試合会場の天気予報を取得し、
            選手の疲労度・コンディションへの影響スコアを計算する
"""

from __future__ import annotations

import logging
from datetime import datetime, date

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL         = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 10


# ─────────────────────────────────────────────
# 天気取得
# ─────────────────────────────────────────────

def get_weather_forecast(lat: float, lon: float, match_date: str | date) -> dict:
    """
    Open-Meteo から試合当日の天気予報を取得。

    Parameters
    ----------
    lat, lon    : スタジアム緯度・経度
    match_date  : "YYYY-MM-DD" 文字列 または date オブジェクト

    Returns
    -------
    dict: {
      "date": str,
      "temp_max": float,   # 最高気温 (°C)
      "temp_min": float,   # 最低気温 (°C)
      "temp_avg": float,   # 平均気温 (°C)
      "precipitation": float,  # 降水量 (mm)
      "wind_speed": float,     # 最大風速 (km/h)
      "weather_code": int,
      "description": str,      # 天気説明（日本語）
      "condition": str,        # "good" / "moderate" / "bad"
      "fatigue_factor": float, # 0.0〜1.0（高いほど疲労増大）
    }
    """
    if isinstance(match_date, date):
        date_str = match_date.isoformat()
    else:
        date_str = str(match_date)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode",
        ],
        "timezone": "Asia/Tokyo",
        "start_date": date_str,
        "end_date": date_str,
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        temp_max  = _first(daily, "temperature_2m_max",  25.0)
        temp_min  = _first(daily, "temperature_2m_min",  18.0)
        temp_avg  = round((temp_max + temp_min) / 2, 1)
        precip    = _first(daily, "precipitation_sum",   0.0)
        wind      = _first(daily, "windspeed_10m_max",   10.0)
        wcode     = int(_first(daily, "weathercode",     0))

        description = _wmo_to_japanese(wcode)
        condition   = _rate_condition(temp_avg, precip, wind)
        fatigue     = _calc_fatigue(temp_avg, precip, wind)

        return {
            "date":          date_str,
            "temp_max":      temp_max,
            "temp_min":      temp_min,
            "temp_avg":      temp_avg,
            "precipitation": precip,
            "wind_speed":    wind,
            "weather_code":  wcode,
            "description":   description,
            "condition":     condition,
            "fatigue_factor": fatigue,
        }

    except Exception as exc:
        logger.warning("Open-Meteo fetch failed: %s", exc)
        return _fallback_weather(date_str)


def get_historical_weather(lat: float, lon: float, date_str: str) -> dict:
    """
    Open-Meteo アーカイブ API で過去の実績天気データを取得。
    予報ではなく実際の観測値を使用する。
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode",
        ],
        "timezone":   "Asia/Tokyo",
        "start_date": date_str,
        "end_date":   date_str,
    }
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        temp_max = _first(daily, "temperature_2m_max", 12.0)
        temp_min = _first(daily, "temperature_2m_min",  5.0)
        temp_avg = round((temp_max + temp_min) / 2, 1)
        precip   = _first(daily, "precipitation_sum",   0.0)
        wind     = _first(daily, "windspeed_10m_max",  10.0)
        wcode    = int(_first(daily, "weathercode",      0))

        description = _wmo_to_japanese(wcode)
        condition   = _rate_condition(temp_avg, precip, wind)
        fatigue     = _calc_fatigue(temp_avg, precip, wind)

        return {
            "date":           date_str,
            "temp_max":       temp_max,
            "temp_min":       temp_min,
            "temp_avg":       temp_avg,
            "precipitation":  precip,
            "wind_speed":     wind,
            "weather_code":   wcode,
            "description":    f"[実績] {description}",
            "condition":      condition,
            "fatigue_factor": fatigue,
        }
    except Exception as exc:
        logger.warning("Open-Meteo archive failed for %s: %s", date_str, exc)
        return _fallback_weather(date_str)


def _first(d: dict, key: str, default: float) -> float:
    lst = d.get(key, [])
    if lst and lst[0] is not None:
        return round(float(lst[0]), 1)
    return default


def _fallback_weather(date_str: str) -> dict:
    return {
        "date": date_str,
        "temp_max": 22.0,
        "temp_min": 15.0,
        "temp_avg": 18.5,
        "precipitation": 0.0,
        "wind_speed": 8.0,
        "weather_code": 0,
        "description": "取得失敗（快晴と仮定）",
        "condition": "good",
        "fatigue_factor": 0.1,
    }


# ─────────────────────────────────────────────
# 疲労度・コンディション計算
# ─────────────────────────────────────────────

def _calc_fatigue(temp_avg: float, precip: float, wind: float) -> float:
    """
    環境要因から疲労増大スコアを計算 (0.0〜1.0)。
    科学的根拠:
      - 高温・多湿は熱疲労を促進（WGBT 研究）
      - 強雨は運動効率を最大 8% 低下（SportsMed 2018）
      - 風速 > 30km/h はスプリント速度に影響（J Sport Sci 2020）
    """
    score = 0.0

    # 気温影響
    if temp_avg >= 32:
        score += 0.40   # 酷暑
    elif temp_avg >= 28:
        score += 0.25   # 真夏日
    elif temp_avg >= 24:
        score += 0.12   # 夏日
    elif temp_avg <= 3:
        score += 0.20   # 極寒（筋収縮効率低下）
    elif temp_avg <= 10:
        score += 0.10   # 寒冷

    # 降水影響
    if precip >= 20:
        score += 0.30   # 大雨
    elif precip >= 10:
        score += 0.18   # 強雨
    elif precip >= 3:
        score += 0.08   # 小雨

    # 強風影響
    if wind >= 50:
        score += 0.25
    elif wind >= 30:
        score += 0.15
    elif wind >= 20:
        score += 0.05

    return round(min(score, 1.0), 3)


def _rate_condition(temp: float, precip: float, wind: float) -> str:
    fatigue = _calc_fatigue(temp, precip, wind)
    if fatigue <= 0.15:
        return "good"
    elif fatigue <= 0.35:
        return "moderate"
    else:
        return "bad"


# ─────────────────────────────────────────────
# WMO 天気コード -> 日本語説明
# ─────────────────────────────────────────────

_WMO_MAP: dict[int, str] = {
    0: "快晴", 1: "ほぼ快晴", 2: "一部曇り", 3: "曇り",
    45: "霧", 48: "霧（着氷性）",
    51: "小雨（霧雨）", 53: "霧雨", 55: "強い霧雨",
    61: "小雨", 63: "雨", 65: "大雨",
    71: "小雪", 73: "雪", 75: "大雪",
    80: "にわか雨", 81: "雨", 82: "激しいにわか雨",
    85: "にわか雪", 86: "大雪（にわか）",
    95: "雷雨", 96: "雷雨（ひょうあり）", 99: "激しい雷雨",
}


def _wmo_to_japanese(code: int) -> str:
    return _WMO_MAP.get(code, f"コード {code}")


# ─────────────────────────────────────────────
# 天気アイコン (Emoji)
# ─────────────────────────────────────────────

def weather_emoji(code: int) -> str:
    if code == 0:       return "☀️"
    if code <= 3:       return "⛅"
    if code <= 48:      return "🌫️"
    if code <= 67:      return "🌧️"
    if code <= 77:      return "🌨️"
    if code <= 82:      return "🌦️"
    if code <= 86:      return "❄️"
    return "⛈️"


def condition_color(condition: str) -> str:
    return {"good": "#27ae60", "moderate": "#f39c12", "bad": "#e74c3c"}.get(condition, "#888")
