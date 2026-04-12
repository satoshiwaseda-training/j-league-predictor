"""
environment_fetch.py - 試合環境データの取得

責務:
  1. Jリーグ公式試合記録ページから天候・気温・湿度・入場者数・ピッチ状態を抽出
  2. Open-Meteo (実績/予報) からスタジアム座標ベースの気象データを取得
  3. スタジアム→座標の対応表管理

データリーク防止:
  - 過去試合: 実績値 (Archive API / 公式記録) を使用
  - 未来試合: 予報値のみ使用し、is_forecast=True フラグで区別
  - 予測対象試合にはキックオフ時点で取得可能な情報のみ使う

対象: J1 / J2 のみ (J3 は対象外)
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, date as _date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
ENV_CSV_PATH = DATA_DIR / "environment_features.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
}
JLEAGUE_BASE = "https://www.jleague.jp"
JDATA_BASE = "https://data.j-league.or.jp"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 12


# ─── 天候・ピッチ状態の正規化辞書 ──────────────────────────

_WEATHER_NORMALIZE: dict[str, str] = {
    "晴": "晴", "晴れ": "晴", "快晴": "晴", "薄晴": "晴",
    "曇": "曇", "曇り": "曇", "くもり": "曇", "薄曇": "曇", "薄曇り": "曇",
    "雨": "雨", "小雨": "雨", "弱雨": "雨", "大雨": "雨", "豪雨": "雨",
    "雪": "雪", "小雪": "雪", "みぞれ": "雪", "霙": "雪",
    "晴のち曇": "晴", "曇のち晴": "曇", "曇のち雨": "曇",
    "晴時々曇": "晴", "曇時々晴": "曇", "曇時々雨": "曇",
    "雨のち曇": "雨", "雨のち晴": "雨",
    "晴/曇": "晴", "曇/晴": "曇", "曇/雨": "曇", "雨/曇": "雨",
}

_PITCH_NORMALIZE: dict[str, str] = {
    "良芝": "良芝", "全面良芝": "良芝", "良": "良芝",
    "乾燥": "乾燥", "やや乾燥": "乾燥",
    "水含み": "水含み", "湿り": "水含み", "やや水含み": "水含み",
    "不良": "不良", "荒れ": "不良",
    "全面人工芝": "人工芝", "人工芝": "人工芝",
}


def normalize_weather(raw: str | None) -> str | None:
    """天候を 晴/曇/雨/雪/その他 に正規化"""
    if not raw:
        return None
    raw = raw.strip()
    if raw in _WEATHER_NORMALIZE:
        return _WEATHER_NORMALIZE[raw]
    # 部分一致で探す
    for key, val in _WEATHER_NORMALIZE.items():
        if key in raw:
            return val
    return "その他"


def normalize_pitch(raw: str | None) -> str | None:
    """ピッチ状態を正規化"""
    if not raw:
        return None
    raw = raw.strip()
    if raw in _PITCH_NORMALIZE:
        return _PITCH_NORMALIZE[raw]
    for key, val in _PITCH_NORMALIZE.items():
        if key in raw:
            return val
    return raw  # 不明なら生値のまま


# ─── 1. Jリーグ公式試合記録から環境情報抽出 ──────────────

def fetch_official_match_environment(match_row: dict) -> dict:
    """
    Jリーグ公式試合ページ / data.j-league.or.jp から
    天候・気温・湿度・入場者数・スタジアム・ピッチ状態を抽出。

    Parameters
    ----------
    match_row : dict with keys:
        date, home_team, away_team, competition, round (optional),
        venue (optional), match_url (optional)

    Returns
    -------
    dict: {
        weather_raw, weather, temperature_c, humidity_pct,
        attendance, stadium, pitch_condition_raw, pitch_condition,
        kickoff_time, source, fetched_at
    }
    """
    result = {
        "weather_raw": None,
        "weather": None,
        "temperature_c": None,
        "humidity_pct": None,
        "attendance": None,
        "stadium": None,
        "pitch_condition_raw": None,
        "pitch_condition": None,
        "kickoff_time": None,
        "source": None,
        "fetched_at": datetime.now().isoformat(),
    }

    # data.j-league.or.jp の試合記録ページを試す
    env = _try_jdata_match_record(match_row)
    if env:
        result.update(env)
        result["source"] = "data.j-league.or.jp"
        return result

    # fallback: jleague.jp の節ページからスタジアム情報を取る
    venue = match_row.get("venue", "")
    if venue:
        result["stadium"] = venue
        result["source"] = "venues.py"

    kickoff = match_row.get("time", "")
    if kickoff and kickoff != "未定":
        result["kickoff_time"] = kickoff

    return result


def _try_jdata_match_record(match_row: dict) -> dict | None:
    """
    data.j-league.or.jp/SFMS01/search → 試合記録ページの環境データ取得。
    公式記録には 天候, 気温, 湿度, 入場者数, ピッチ が含まれる。
    """
    date_str = match_row.get("date", "")
    home = match_row.get("home_team", "")
    away = match_row.get("away_team", "")
    competition = match_row.get("competition", "").upper()

    if not date_str or not home:
        return None

    # 試合記録ページの検索URLを構築
    year = date_str[:4]
    comp_id = "460" if "J1" in competition else "461"  # J1=460, J2=461

    search_url = (
        f"{JDATA_BASE}/SFMS01/search?"
        f"competition_years={year}"
        f"&competition_frame_ids={comp_id}"
        f"&tv_relay_station_name="
    )

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        html = resp.content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        # 試合リストから該当試合のリンクを探す
        match_link = _find_match_link(soup, date_str, home, away)
        if not match_link:
            return None

        # 個別試合記録ページを取得
        time.sleep(0.5)  # 礼儀的ウェイト
        detail_url = JDATA_BASE + match_link if match_link.startswith("/") else match_link
        resp2 = requests.get(detail_url, headers=HEADERS, timeout=TIMEOUT)
        resp2.raise_for_status()
        detail_html = resp2.content.decode("utf-8", errors="ignore")
        detail_soup = BeautifulSoup(detail_html, "lxml")

        return _parse_match_record_page(detail_soup)

    except Exception as e:
        logger.debug("jdata試合記録取得失敗: %s", e)
        return None


def _find_match_link(soup: BeautifulSoup, date_str: str, home: str, away: str) -> str | None:
    """試合リストから date + team で一致するリンクを探す"""
    import unicodedata

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).strip()

    target_date_parts = date_str.split("-")
    if len(target_date_parts) == 3:
        target_month_day = f"{int(target_date_parts[1])}月{int(target_date_parts[2])}日"
    else:
        return None

    home_norm = _norm(home)

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/SFMS02/" not in href and "/SFMS01/" not in href:
            continue
        text = _norm(link.get_text())
        # チーム名の短縮形でもマッチ可能にする
        parent_text = _norm(link.parent.get_text()) if link.parent else ""
        combined = text + " " + parent_text

        if home_norm[:3] in combined or home_norm[:4] in combined:
            if target_month_day in combined or date_str in combined:
                return href

    return None


def _parse_match_record_page(soup: BeautifulSoup) -> dict | None:
    """試合記録ページから環境データを抽出"""
    result = {}

    # テーブルから key-value を収集
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)

            if "天候" in label or "天気" in label:
                result["weather_raw"] = value
                result["weather"] = normalize_weather(value)
            elif "気温" in label:
                temp = _parse_number(value)
                if temp is not None:
                    result["temperature_c"] = temp
            elif "湿度" in label:
                hum = _parse_number(value)
                if hum is not None:
                    result["humidity_pct"] = hum
            elif "入場者数" in label or "観客" in label or "入場" in label:
                att = _parse_attendance(value)
                if att is not None:
                    result["attendance"] = att
            elif "スタジアム" in label or "会場" in label or "競技場" in label:
                result["stadium"] = value
            elif "ピッチ" in label or "芝" in label:
                result["pitch_condition_raw"] = value
                result["pitch_condition"] = normalize_pitch(value)
            elif "キックオフ" in label or "開始" in label:
                time_m = re.search(r"(\d{1,2}:\d{2})", value)
                if time_m:
                    result["kickoff_time"] = time_m.group(1)

    return result if result else None


def _parse_number(s: str) -> float | None:
    """文字列から数値を抽出 ('23.5℃' → 23.5, '67%' → 67)"""
    m = re.search(r"[\d.]+", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _parse_attendance(s: str) -> int | None:
    """入場者数文字列をパース ('12,345人' → 12345)"""
    cleaned = re.sub(r"[^\d]", "", s)
    if cleaned:
        try:
            return int(cleaned)
        except ValueError:
            pass
    return None


# ─── 2. Open-Meteo からスタジアム座標ベースの気象データ取得 ─

def fetch_weather_observation_for_stadium(
    match_row: dict,
    use_archive: bool | None = None,
) -> dict:
    """
    スタジアム座標ベースで気象データを取得。

    Parameters
    ----------
    match_row : dict with keys: date, home_team, venue (optional)
    use_archive : None=自動判定, True=実績, False=予報

    Returns
    -------
    dict: {
        temperature_c, humidity_pct, wind_speed_kmh, precipitation_mm,
        weather_code, weather, wbgt_estimate, is_forecast,
        source, fetched_at
    }
    """
    from venues import get_venue_info

    result = {
        "temperature_c": None,
        "humidity_pct": None,
        "wind_speed_kmh": None,
        "precipitation_mm": None,
        "weather_code": None,
        "weather": None,
        "wbgt_estimate": None,
        "is_forecast": None,
        "source": None,
        "fetched_at": datetime.now().isoformat(),
    }

    home = match_row.get("home_team", "")
    venue_name = match_row.get("venue", "")
    venue = get_venue_info(home, venue_name)
    lat, lon = venue.get("lat"), venue.get("lon")
    if lat is None or lon is None:
        return result

    date_str = match_row.get("date", "")
    if not date_str:
        return result

    # 過去/未来の自動判定
    if use_archive is None:
        try:
            match_date = _date.fromisoformat(date_str)
            use_archive = match_date < _date.today()
        except ValueError:
            use_archive = False

    result["is_forecast"] = not use_archive

    # キックオフ時刻があれば時間帯指定
    kickoff = match_row.get("time", match_row.get("kickoff_time", ""))
    kickoff_hour = None
    if kickoff and kickoff != "未定":
        m = re.search(r"(\d{1,2}):", kickoff)
        if m:
            kickoff_hour = int(m.group(1))

    try:
        if use_archive:
            data = _fetch_open_meteo_hourly_archive(lat, lon, date_str, kickoff_hour)
        else:
            data = _fetch_open_meteo_hourly_forecast(lat, lon, date_str, kickoff_hour)

        if data:
            result.update(data)
            result["source"] = "open-meteo-archive" if use_archive else "open-meteo-forecast"
            # WBGT 推定 (Liljegren simplified)
            if result["temperature_c"] is not None and result["humidity_pct"] is not None:
                result["wbgt_estimate"] = _estimate_wbgt(
                    result["temperature_c"],
                    result["humidity_pct"],
                    result.get("wind_speed_kmh", 10),
                )

    except Exception as e:
        logger.warning("Open-Meteo取得失敗 (%s): %s", date_str, e)

    return result


def _fetch_open_meteo_hourly_archive(
    lat: float, lon: float, date_str: str, kickoff_hour: int | None,
) -> dict | None:
    """Open-Meteo Archive API から時間別実績データを取得"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "windspeed_10m",
            "precipitation",
            "weathercode",
        ],
        "timezone": "Asia/Tokyo",
        "start_date": date_str,
        "end_date": date_str,
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    return _extract_hourly_at_kickoff(hourly, kickoff_hour)


def _fetch_open_meteo_hourly_forecast(
    lat: float, lon: float, date_str: str, kickoff_hour: int | None,
) -> dict | None:
    """Open-Meteo Forecast API から時間別予報データを取得"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "windspeed_10m",
            "precipitation",
            "weathercode",
        ],
        "timezone": "Asia/Tokyo",
        "start_date": date_str,
        "end_date": date_str,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    return _extract_hourly_at_kickoff(hourly, kickoff_hour)


def _extract_hourly_at_kickoff(hourly: dict, kickoff_hour: int | None) -> dict | None:
    """時間別データからキックオフ時刻付近のデータを取得"""
    times = hourly.get("time", [])
    if not times:
        return None

    # キックオフ時刻のインデックスを特定 (デフォルト15時)
    target_hour = kickoff_hour if kickoff_hour is not None else 15
    target_idx = min(target_hour, len(times) - 1)

    # 時刻文字列からhourを抽出してマッチ
    for i, t in enumerate(times):
        m = re.search(r"T(\d{2}):", t)
        if m and int(m.group(1)) == target_hour:
            target_idx = i
            break

    def _val(key: str) -> float | None:
        vals = hourly.get(key, [])
        if target_idx < len(vals) and vals[target_idx] is not None:
            return round(float(vals[target_idx]), 1)
        return None

    temp = _val("temperature_2m")
    humidity = _val("relative_humidity_2m")
    wind = _val("windspeed_10m")
    precip = _val("precipitation")
    wcode = hourly.get("weathercode", [])
    weather_code = int(wcode[target_idx]) if target_idx < len(wcode) and wcode[target_idx] is not None else None

    # WMO code → 正規化天候
    weather = None
    if weather_code is not None:
        weather = _wmo_to_category(weather_code)

    return {
        "temperature_c": temp,
        "humidity_pct": humidity,
        "wind_speed_kmh": wind,
        "precipitation_mm": precip,
        "weather_code": weather_code,
        "weather": weather,
    }


def _wmo_to_category(code: int) -> str:
    """WMO weather code → 正規化カテゴリ"""
    if code <= 1:
        return "晴"
    elif code <= 3:
        return "曇"
    elif code <= 48:
        return "曇"  # 霧
    elif code <= 67:
        return "雨"
    elif code <= 77:
        return "雪"
    elif code <= 82:
        return "雨"  # にわか雨
    elif code <= 86:
        return "雪"
    else:
        return "雨"  # 雷雨


def _estimate_wbgt(temp_c: float, humidity_pct: float, wind_kmh: float = 10) -> float:
    """
    WBGT 簡易推定 (Liljegren simplified approximation)。
    屋外日なたを想定。正確なWBGTには黒球温度が必要だが、
    気温と湿度から実用的な近似値を算出。

    参考: 環境省 熱中症予防情報 / ACSM Position Stand
    """
    # Steadman の近似
    # WBGT ≈ 0.567 * T + 0.393 * e + 3.94
    # e = (humidity/100) * 6.105 * exp(17.27 * T / (237.7 + T))
    e = (humidity_pct / 100) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    wbgt = 0.567 * temp_c + 0.393 * e + 3.94

    # 風速補正 (風が強いと体感的にWBGTが下がる)
    wind_ms = wind_kmh / 3.6
    if wind_ms > 3:
        wbgt -= min((wind_ms - 3) * 0.3, 2.0)

    return round(max(wbgt, 0), 1)


# ─── 3. スタジアム→座標 対応表 (環境マスター) ─────────────

def build_environment_master() -> pd.DataFrame:
    """
    venues.py のスタジアムDB + チーム→ホーム会場 対応を
    DataFrame 形式でエクスポート。
    将来的に nearest weather station マッピングを追加する基盤。
    """
    from venues import J_LEAGUE_VENUES, TEAM_HOME_VENUES

    rows = []
    for team, venue_name in TEAM_HOME_VENUES.items():
        venue = J_LEAGUE_VENUES.get(venue_name, {})
        rows.append({
            "team": team,
            "stadium": venue_name,
            "lat": venue.get("lat"),
            "lon": venue.get("lon"),
            "city": venue.get("city", ""),
            "capacity": venue.get("capacity", 0),
        })

    return pd.DataFrame(rows)


# ─── 4. バッチ取得 ────────────────────────────────────────

def fetch_environment_for_matches(
    df_matches: pd.DataFrame,
    rate_limit_sec: float = 0.3,
) -> pd.DataFrame:
    """
    複数試合の環境データをバッチ取得。

    Parameters
    ----------
    df_matches : match_id, date, home_team, away_team, competition,
                 venue (optional), time (optional) を含む DataFrame
    rate_limit_sec : API呼び出し間隔

    Returns
    -------
    DataFrame: 環境データ (match_id をキーに結合可能)
    """
    rows: list[dict] = []

    for idx, match in df_matches.iterrows():
        match_dict = match.to_dict()
        match_id = match_dict.get("match_id", "")

        logger.debug("環境データ取得: %s (%s vs %s)",
                     match_dict.get("date"), match_dict.get("home_team"), match_dict.get("away_team"))

        # 公式記録からの取得
        official = fetch_official_match_environment(match_dict)

        # Open-Meteo からの取得
        weather_obs = fetch_weather_observation_for_stadium(match_dict)

        # マージ (公式優先、Open-Meteo で補完)
        env = {"match_id": match_id}

        # 天候
        env["weather_raw"] = official.get("weather_raw")
        env["weather"] = official.get("weather") or weather_obs.get("weather")

        # 気温: 公式 > Open-Meteo
        env["temperature_c"] = official.get("temperature_c") or weather_obs.get("temperature_c")

        # 湿度: 公式 > Open-Meteo
        env["humidity_pct"] = official.get("humidity_pct") or weather_obs.get("humidity_pct")

        # 風速・降水量: Open-Meteo のみ
        env["wind_speed_kmh"] = weather_obs.get("wind_speed_kmh")
        env["precipitation_mm"] = weather_obs.get("precipitation_mm")

        # WBGT: Open-Meteo 推定
        env["wbgt_estimate"] = weather_obs.get("wbgt_estimate")

        # 入場者数: 公式のみ
        env["attendance"] = official.get("attendance")

        # スタジアム
        env["stadium"] = official.get("stadium") or match_dict.get("venue", "")

        # ピッチ状態
        env["pitch_condition_raw"] = official.get("pitch_condition_raw")
        env["pitch_condition"] = official.get("pitch_condition")

        # キックオフ時刻
        env["kickoff_time"] = official.get("kickoff_time") or match_dict.get("time", "")

        # メタ
        env["weather_code"] = weather_obs.get("weather_code")
        env["is_forecast"] = weather_obs.get("is_forecast", False)
        env["official_source"] = official.get("source", "")
        env["weather_source"] = weather_obs.get("source", "")
        env["fetched_at"] = datetime.now().isoformat()

        rows.append(env)

        if rate_limit_sec > 0:
            time.sleep(rate_limit_sec)

    return pd.DataFrame(rows)


# ─── 5. 保存・読込 ────────────────────────────────────────

def save_environment_data(df: pd.DataFrame) -> Path:
    """環境データを CSV に保存 (match_id で上書き更新)"""
    if df.empty:
        return ENV_CSV_PATH

    existing = load_environment_data()

    if not existing.empty and "match_id" in existing.columns and "match_id" in df.columns:
        # 既存データから新規データの match_id を除外してマージ
        new_ids = set(df["match_id"].dropna())
        kept = existing[~existing["match_id"].isin(new_ids)]
        merged = pd.concat([kept, df], ignore_index=True)
    else:
        merged = df

    if not merged.empty and "match_id" in merged.columns:
        merged = merged.sort_values("match_id").reset_index(drop=True)

    merged.to_csv(ENV_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("環境データ保存: %s (%d行)", ENV_CSV_PATH, len(merged))
    return ENV_CSV_PATH


def load_environment_data() -> pd.DataFrame:
    """保存済み環境データを読み込む"""
    if ENV_CSV_PATH.exists():
        try:
            return pd.read_csv(ENV_CSV_PATH, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(ENV_CSV_PATH, encoding="utf-8")
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()
