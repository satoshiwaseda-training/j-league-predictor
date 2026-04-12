"""
environment_features.py - 試合環境データからの特徴量生成

責務:
  1. 環境特徴量 (天候・気温・湿度・キックオフ等) の生成
  2. 疲労・移動特徴量 (休養日数・移動距離・連戦フラグ) の生成
  3. 交互作用特徴量 (hot*travel, wbgt*congested 等)
  4. match_id での安全な結合

設計方針:
  - 欠損は None のまま保持し、無理に埋めない
  - categorical は正規化辞書を通す
  - 後から再取得・再計算できる構造
  - ABテスト用にレベル別の特徴量セットを定義
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, date as _date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ENV_FEATURES_PATH = DATA_DIR / "environment_features.csv"


# ─── 閾値定義 ────────────────────────────────────────────

HIGH_TEMP_THRESHOLD = 28.0       # 真夏日
EXTREME_TEMP_THRESHOLD = 33.0    # 猛暑日
HIGH_HUMIDITY_THRESHOLD = 75.0   # 高湿度
HOT_HUMID_TEMP = 26.0            # 高温多湿の温度下限
HOT_HUMID_HUMIDITY = 70.0        # 高温多湿の湿度下限
WBGT_WARNING_THRESHOLD = 25.0    # WBGT 警戒
WBGT_DANGER_THRESHOLD = 28.0     # WBGT 厳重警戒
WBGT_CRITICAL_THRESHOLD = 31.0   # WBGT 危険
RAIN_THRESHOLD_MM = 1.0          # 雨判定 (降水量mm)
HEAVY_RAIN_THRESHOLD_MM = 10.0   # 大雨判定
STRONG_WIND_THRESHOLD_KMH = 25.0 # 強風判定
DAY_GAME_BEFORE = 16             # デーゲーム判定 (16時キックオフ未満)
NIGHT_GAME_AFTER = 18            # ナイター判定 (18時キックオフ以降)
CONGESTED_DAYS = 3               # 連戦判定 (3日以内)
LONG_TRIP_KM = 600               # 長距離移動判定


# ─── 1. 試合環境特徴量の生成 ─────────────────────────────

def build_match_environment_features(df_env: pd.DataFrame) -> pd.DataFrame:
    """
    環境データ DataFrame から試合環境特徴量を生成。

    Parameters
    ----------
    df_env : environment_fetch.fetch_environment_for_matches() の出力。
             必須列: match_id
             利用列: temperature_c, humidity_pct, weather, kickoff_time,
                     pitch_condition, attendance, wind_speed_kmh,
                     precipitation_mm, wbgt_estimate

    Returns
    -------
    DataFrame with environment feature columns appended
    """
    if df_env.empty:
        return df_env

    df = df_env.copy()

    # --- キックオフ時刻特徴量 ---
    df["kickoff_hour"] = df.apply(_extract_kickoff_hour, axis=1)
    df["is_day_game"] = df["kickoff_hour"].apply(
        lambda h: True if pd.notna(h) and h < DAY_GAME_BEFORE else
                  (False if pd.notna(h) else None)
    )
    df["is_night_game"] = df["kickoff_hour"].apply(
        lambda h: True if pd.notna(h) and h >= NIGHT_GAME_AFTER else
                  (False if pd.notna(h) else None)
    )

    # --- 天候カテゴリ特徴量 ---
    df["weather_category"] = df.get("weather", pd.Series(dtype="object"))

    # --- 気温特徴量 ---
    df["high_temp_flag"] = df["temperature_c"].apply(
        lambda t: True if pd.notna(t) and t >= HIGH_TEMP_THRESHOLD else
                  (False if pd.notna(t) else None)
    )
    df["extreme_temp_flag"] = df["temperature_c"].apply(
        lambda t: True if pd.notna(t) and t >= EXTREME_TEMP_THRESHOLD else
                  (False if pd.notna(t) else None)
    )

    # --- 湿度特徴量 ---
    df["high_humidity_flag"] = df["humidity_pct"].apply(
        lambda h: True if pd.notna(h) and h >= HIGH_HUMIDITY_THRESHOLD else
                  (False if pd.notna(h) else None)
    )

    # --- 高温多湿フラグ ---
    df["hot_humid_flag"] = df.apply(
        lambda r: (
            True if (pd.notna(r.get("temperature_c")) and pd.notna(r.get("humidity_pct"))
                     and r["temperature_c"] >= HOT_HUMID_TEMP
                     and r["humidity_pct"] >= HOT_HUMID_HUMIDITY)
            else (False if pd.notna(r.get("temperature_c")) and pd.notna(r.get("humidity_pct"))
                  else None)
        ), axis=1,
    )

    # --- WBGT フラグ ---
    df["wbgt_warning_flag"] = df.get("wbgt_estimate", pd.Series(dtype="float")).apply(
        lambda w: True if pd.notna(w) and w >= WBGT_WARNING_THRESHOLD else
                  (False if pd.notna(w) else None)
    )
    df["wbgt_danger_flag"] = df.get("wbgt_estimate", pd.Series(dtype="float")).apply(
        lambda w: True if pd.notna(w) and w >= WBGT_DANGER_THRESHOLD else
                  (False if pd.notna(w) else None)
    )

    # --- 降水フラグ ---
    df["rain_flag"] = df.apply(_compute_rain_flag, axis=1)
    df["heavy_rain_flag"] = df.get("precipitation_mm", pd.Series(dtype="float")).apply(
        lambda p: True if pd.notna(p) and p >= HEAVY_RAIN_THRESHOLD_MM else
                  (False if pd.notna(p) else None)
    )

    # --- 強風フラグ ---
    df["strong_wind_flag"] = df.get("wind_speed_kmh", pd.Series(dtype="float")).apply(
        lambda w: True if pd.notna(w) and w >= STRONG_WIND_THRESHOLD_KMH else
                  (False if pd.notna(w) else None)
    )

    # --- 夏季ウィンドウフラグ (7-8月) ---
    df["summer_window_flag"] = df.get("match_id", pd.Series(dtype="object")).apply(
        _is_summer_window
    )

    # --- ピッチ状態バッドフラグ ---
    df["pitch_condition_bad_flag"] = df.get("pitch_condition", pd.Series(dtype="object")).apply(
        lambda p: True if pd.notna(p) and p in ("水含み", "不良") else
                  (False if pd.notna(p) else None)
    )

    # --- クラブ規模/環境 proxy (nullable、後段で追加) ---
    df["club_scale_proxy"] = None
    df["training_facility_proxy"] = None
    df["squad_depth_proxy"] = None

    return df


def _extract_kickoff_hour(row) -> int | None:
    """kickoff_time 文字列からhourを抽出"""
    kt = row.get("kickoff_time", "")
    if not kt or kt == "未定":
        return None
    import re
    m = re.search(r"(\d{1,2}):", str(kt))
    if m:
        return int(m.group(1))
    return None


def _compute_rain_flag(row) -> bool | None:
    """降水量 or 天候カテゴリから雨判定"""
    precip = row.get("precipitation_mm")
    if pd.notna(precip) and precip >= RAIN_THRESHOLD_MM:
        return True
    weather = row.get("weather")
    if pd.notna(weather) and weather == "雨":
        return True
    if pd.notna(precip):
        return False
    if pd.notna(weather):
        return False
    return None


def _is_summer_window(match_id) -> bool | None:
    """match_id から夏季ウィンドウ (7-8月) を判定"""
    if not match_id or not isinstance(match_id, str):
        return None
    parts = str(match_id).split("-")
    if len(parts) >= 2:
        try:
            month = int(parts[1].split("_")[0]) if "_" in parts[1] else int(parts[1])
            return month in (7, 8)
        except (ValueError, IndexError):
            pass
    return None


# ─── 2. 疲労・移動特徴量の生成 ───────────────────────────

def build_fatigue_travel_features(
    df_matches: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    試合一覧と履歴から休養日数・移動距離・連戦フラグを算出。

    Parameters
    ----------
    df_matches : 特徴量を付与したい試合一覧
                 必須列: match_id, date, home_team, away_team
    history_df : 過去の試合結果 (date, home_team, away_team を含む)
                 None なら weekend_update.load_history() から読み込み

    Returns
    -------
    DataFrame with fatigue/travel columns appended
    """
    from scripts.predict_logic import haversine_km
    from venues import get_venue_info

    if df_matches.empty:
        return df_matches

    if history_df is None:
        from weekend_update import load_history
        history_df = load_history()

    df = df_matches.copy()

    # 履歴からチームごとの試合日リストを構築 (データリーク防止: 対象試合より前のみ)
    team_dates: dict[str, list[str]] = {}
    if not history_df.empty:
        for _, row in history_df.iterrows():
            d = str(row.get("date", ""))
            for col in ("home_team", "away_team"):
                team = str(row.get(col, ""))
                if team and d:
                    team_dates.setdefault(team, []).append(d)
        # ソート
        for team in team_dates:
            team_dates[team] = sorted(set(team_dates[team]))

    # 各試合について特徴量を計算
    days_rest_home_list = []
    days_rest_away_list = []
    travel_km_list = []
    long_trip_list = []
    congested_home_list = []
    congested_away_list = []

    for _, row in df.iterrows():
        match_date = str(row.get("date", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        # --- 休養日数 ---
        days_h = _calc_days_since_last(home, match_date, team_dates)
        days_a = _calc_days_since_last(away, match_date, team_dates)
        days_rest_home_list.append(days_h)
        days_rest_away_list.append(days_a)

        # --- 連戦フラグ ---
        congested_home_list.append(
            True if days_h is not None and days_h <= CONGESTED_DAYS else
            (False if days_h is not None else None)
        )
        congested_away_list.append(
            True if days_a is not None and days_a <= CONGESTED_DAYS else
            (False if days_a is not None else None)
        )

        # --- アウェイ移動距離 ---
        try:
            home_venue = get_venue_info(home)
            away_venue = get_venue_info(away)
            dist = haversine_km(
                away_venue["lat"], away_venue["lon"],
                home_venue["lat"], home_venue["lon"],
            )
            travel_km_list.append(dist)
            long_trip_list.append(dist >= LONG_TRIP_KM)
        except Exception:
            travel_km_list.append(None)
            long_trip_list.append(None)

    df["days_rest_home"] = days_rest_home_list
    df["days_rest_away"] = days_rest_away_list
    df["away_travel_distance_km"] = travel_km_list
    df["away_long_trip_flag"] = long_trip_list
    df["congested_schedule_home"] = congested_home_list
    df["congested_schedule_away"] = congested_away_list

    # 統合連戦フラグ (どちらかが連戦)
    df["congested_schedule_flag"] = df.apply(
        lambda r: (
            True if r.get("congested_schedule_home") is True or r.get("congested_schedule_away") is True
            else (False if r.get("congested_schedule_home") is not None else None)
        ), axis=1,
    )

    return df


def _calc_days_since_last(
    team: str,
    match_date: str,
    team_dates: dict[str, list[str]],
) -> int | None:
    """チームの前回試合からの日数を算出 (データリーク防止: match_date より前のみ参照)"""
    dates = team_dates.get(team, [])
    if not dates or not match_date:
        return None

    try:
        target = _date.fromisoformat(match_date)
    except ValueError:
        return None

    # match_date より前の日付のみ
    prev_dates = [d for d in dates if d < match_date]
    if not prev_dates:
        return None

    last = prev_dates[-1]  # ソート済みなので最後が直前
    try:
        last_date = _date.fromisoformat(last)
        return (target - last_date).days
    except ValueError:
        return None


# ─── 3. 交互作用特徴量 ───────────────────────────────────

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    環境×疲労の交互作用特徴量を生成。

    生成される特徴量:
      - hot_x_travel: 高温フラグ × アウェイ移動距離 (正規化)
      - wbgt_x_congested: WBGT警戒フラグ × 連戦フラグ
      - rain_x_bad_pitch: 雨フラグ × ピッチ不良フラグ
      - night_x_long_trip: ナイターフラグ × 長距離移動フラグ
    """
    if df.empty:
        return df

    df = df.copy()

    # hot_game_flag * away_travel_distance_km (正規化: 距離/1000)
    df["hot_x_travel"] = df.apply(
        lambda r: (
            (r.get("away_travel_distance_km", 0) or 0) / 1000.0
            if r.get("high_temp_flag") is True else 0.0
        ) if r.get("high_temp_flag") is not None and r.get("away_travel_distance_km") is not None
        else None,
        axis=1,
    )

    # wbgt_warning_flag * congested_schedule_flag
    df["wbgt_x_congested"] = df.apply(
        lambda r: (
            1.0 if r.get("wbgt_warning_flag") is True and r.get("congested_schedule_flag") is True
            else 0.0
        ) if r.get("wbgt_warning_flag") is not None and r.get("congested_schedule_flag") is not None
        else None,
        axis=1,
    )

    # rain_flag * pitch_condition_bad_flag
    df["rain_x_bad_pitch"] = df.apply(
        lambda r: (
            1.0 if r.get("rain_flag") is True and r.get("pitch_condition_bad_flag") is True
            else 0.0
        ) if r.get("rain_flag") is not None and r.get("pitch_condition_bad_flag") is not None
        else None,
        axis=1,
    )

    # night_game_flag * away_long_trip_flag
    df["night_x_long_trip"] = df.apply(
        lambda r: (
            1.0 if r.get("is_night_game") is True and r.get("away_long_trip_flag") is True
            else 0.0
        ) if r.get("is_night_game") is not None and r.get("away_long_trip_flag") is not None
        else None,
        axis=1,
    )

    return df


# ─── 4. 結合関数 ──────────────────────────────────────────

def merge_environment_features(
    match_df: pd.DataFrame,
    env_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    match_id で試合データと環境特徴量を安全に結合。

    - match_id をキーに left join
    - 重複列は env_df 側を優先 (サフィックスなし)
    - match_df に match_id がなければそのまま返す
    """
    if match_df.empty or env_df.empty:
        return match_df

    if "match_id" not in match_df.columns or "match_id" not in env_df.columns:
        logger.warning("match_id カラムが見つかりません。結合をスキップ。")
        return match_df

    # 結合前に env_df から match_id 以外で match_df と重複する列を除去
    overlap_cols = set(match_df.columns) & set(env_df.columns) - {"match_id"}
    env_clean = env_df.drop(columns=list(overlap_cols), errors="ignore")

    merged = match_df.merge(env_clean, on="match_id", how="left")
    return merged


# ─── 5. 統合パイプライン ─────────────────────────────────

def build_full_environment_features(
    df_matches: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    fetch_weather: bool = True,
    rate_limit_sec: float = 0.3,
) -> pd.DataFrame:
    """
    試合一覧に対して全環境特徴量を一括生成。

    Steps:
      1. 環境データ取得 (Open-Meteo / 公式記録)
      2. 環境特徴量生成
      3. 疲労・移動特徴量生成
      4. 交互作用特徴量生成
      5. 保存

    Parameters
    ----------
    df_matches : match_id, date, home_team, away_team 等を含む DataFrame
    history_df : 過去試合の DataFrame (休養日数計算用)
    fetch_weather : True なら API から天候データを取得
    rate_limit_sec : API 呼び出し間隔

    Returns
    -------
    全特徴量を含む DataFrame
    """
    from environment_fetch import (
        fetch_environment_for_matches,
        save_environment_data,
        load_environment_data,
    )

    if df_matches.empty:
        return df_matches

    # Step 1: 環境データ取得
    if fetch_weather:
        env_raw = fetch_environment_for_matches(df_matches, rate_limit_sec)
    else:
        env_raw = load_environment_data()
        if env_raw.empty:
            logger.info("保存済み環境データなし。API取得をスキップ。")
            env_raw = pd.DataFrame({"match_id": df_matches["match_id"]})

    # Step 2: 環境特徴量
    env_features = build_match_environment_features(env_raw)

    # Step 3: 疲労・移動特徴量
    # まず match_df に必要な列を結合
    merge_cols = ["match_id", "date", "home_team", "away_team"]
    available = [c for c in merge_cols if c in df_matches.columns]
    temp_df = df_matches[available].copy()

    if "match_id" in env_features.columns:
        temp_df = temp_df.merge(
            env_features.drop(columns=[c for c in available if c != "match_id"], errors="ignore"),
            on="match_id", how="left",
        )

    fatigue_df = build_fatigue_travel_features(temp_df, history_df)

    # Step 4: 交互作用特徴量
    full_df = build_interaction_features(fatigue_df)

    # Step 5: 保存
    if fetch_weather:
        save_environment_data(env_raw)

    return full_df


# ─── 6. ABテスト用の特徴量セット定義 ─────────────────────

# Level 1: 基本環境 (temperature/humidity/kickoff_hour)
LEVEL1_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "kickoff_hour",
    "is_day_game",
    "is_night_game",
    "weather_category",
    "high_temp_flag",
    "high_humidity_flag",
    "hot_humid_flag",
]

# Level 2: + ピッチ/入場者
LEVEL2_FEATURES = LEVEL1_FEATURES + [
    "pitch_condition",
    "pitch_condition_bad_flag",
    "attendance",
]

# Level 3: + WBGT/移動/休養
LEVEL3_FEATURES = LEVEL2_FEATURES + [
    "wbgt_estimate",
    "wbgt_warning_flag",
    "wbgt_danger_flag",
    "wind_speed_kmh",
    "strong_wind_flag",
    "precipitation_mm",
    "rain_flag",
    "heavy_rain_flag",
    "summer_window_flag",
    "days_rest_home",
    "days_rest_away",
    "away_travel_distance_km",
    "away_long_trip_flag",
    "congested_schedule_flag",
]

# Level 4: + 交互作用
LEVEL4_FEATURES = LEVEL3_FEATURES + [
    "hot_x_travel",
    "wbgt_x_congested",
    "rain_x_bad_pitch",
    "night_x_long_trip",
]

FEATURE_LEVELS = {
    1: LEVEL1_FEATURES,
    2: LEVEL2_FEATURES,
    3: LEVEL3_FEATURES,
    4: LEVEL4_FEATURES,
}


def get_features_for_level(level: int = 1) -> list[str]:
    """ABテストレベルに応じた特徴量リストを返す"""
    return FEATURE_LEVELS.get(level, LEVEL1_FEATURES)


# ─── 7. ABテスト実行 ──────────────────────────────────────

def run_ab_test(
    df_with_features: pd.DataFrame,
    levels: list[int] | None = None,
) -> dict:
    """
    環境特徴量の各レベルで予測精度を比較。

    Parameters
    ----------
    df_with_features : 環境特徴量 + 予測結果 + 実結果 を含む DataFrame
                       必須列: actual_result, pred_prob_h, pred_prob_d, pred_prob_a,
                               pred_winner, is_correct

    Returns
    -------
    dict: {level: {accuracy, macro_f1, logloss, brier, draw_recall, high_conf_miss}}
    """
    if df_with_features.empty:
        return {}

    if levels is None:
        levels = [1, 2, 3, 4]

    results = {}

    # ベースライン (環境特徴量なし)
    results["baseline"] = _compute_metrics(df_with_features, label="baseline")

    for level in levels:
        features = get_features_for_level(level)
        # 特徴量が利用可能な行のみでフィルタ
        available_feats = [f for f in features if f in df_with_features.columns]
        if not available_feats:
            results[f"level_{level}"] = {"note": "特徴量なし"}
            continue

        # 特徴量が全て None でない行のみ (少なくとも1つは値がある)
        mask = df_with_features[available_feats].notna().any(axis=1)
        subset = df_with_features[mask]

        results[f"level_{level}"] = _compute_metrics(
            subset,
            label=f"level_{level}",
            feature_count=len(available_feats),
            sample_count=len(subset),
        )

    return results


def _compute_metrics(df: pd.DataFrame, label: str = "", **extra) -> dict:
    """予測評価メトリクスを計算"""
    import math as _math

    n = len(df)
    if n == 0:
        return {"n": 0, "label": label, **extra}

    # Accuracy
    if "is_correct" in df.columns:
        correct = df["is_correct"].sum()
        accuracy = correct / n if n > 0 else None
    else:
        accuracy = None
        correct = 0

    # Brier Score
    brier_vals = []
    logloss_vals = []
    for _, row in df.iterrows():
        actual = row.get("actual_result", "")
        ph = _safe_float(row.get("pred_prob_h", 33.3))
        pd_ = _safe_float(row.get("pred_prob_d", 33.3))
        pa = _safe_float(row.get("pred_prob_a", 33.3))

        if actual in ("H", "D", "A"):
            # Brier
            p = [ph / 100, pd_ / 100, pa / 100]
            av = [1.0 if actual == "H" else 0.0,
                  1.0 if actual == "D" else 0.0,
                  1.0 if actual == "A" else 0.0]
            brier_vals.append(sum((p[i] - av[i]) ** 2 for i in range(3)))

            # LogLoss
            eps = 1e-10
            probs = {"H": max(ph / 100, eps), "D": max(pd_ / 100, eps), "A": max(pa / 100, eps)}
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            logloss_vals.append(-_math.log(probs.get(actual, eps)))

    avg_brier = sum(brier_vals) / len(brier_vals) if brier_vals else None
    avg_logloss = sum(logloss_vals) / len(logloss_vals) if logloss_vals else None

    # Draw Recall
    actual_draws = df[df.get("actual_result", pd.Series()) == "D"] if "actual_result" in df.columns else pd.DataFrame()
    draw_pred_correct = 0
    if not actual_draws.empty and "pred_winner" in df.columns:
        draw_pred_correct = len(actual_draws[actual_draws["pred_winner"] == "draw"])
    draw_recall = draw_pred_correct / len(actual_draws) if len(actual_draws) > 0 else None

    # 高確信外し
    high_conf_miss = 0
    if "is_correct" in df.columns:
        for _, row in df.iterrows():
            max_prob = max(
                _safe_float(row.get("pred_prob_h", 0)),
                _safe_float(row.get("pred_prob_d", 0)),
                _safe_float(row.get("pred_prob_a", 0)),
            )
            if max_prob >= 60 and row.get("is_correct") is False:
                high_conf_miss += 1

    # Macro F1 (H/D/A)
    macro_f1 = _compute_macro_f1(df) if "actual_result" in df.columns and "pred_winner" in df.columns else None

    return {
        "label": label,
        "n": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "correct": int(correct),
        "avg_brier": round(avg_brier, 4) if avg_brier is not None else None,
        "avg_logloss": round(avg_logloss, 4) if avg_logloss is not None else None,
        "draw_recall": round(draw_recall, 4) if draw_recall is not None else None,
        "high_conf_miss": high_conf_miss,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        **extra,
    }


def _compute_macro_f1(df: pd.DataFrame) -> float | None:
    """3クラス (H/D/A) の Macro F1 を計算"""
    if "actual_result" not in df.columns or "pred_winner" not in df.columns:
        return None

    # pred_winner は "home"/"draw"/"away"、actual_result は "H"/"D"/"A"
    actual_map = {"H": "home", "D": "draw", "A": "away"}

    f1_scores = []
    for cls in ["home", "draw", "away"]:
        tp = len(df[(df["pred_winner"] == cls) & (df["actual_result"].map(actual_map) == cls)])
        fp = len(df[(df["pred_winner"] == cls) & (df["actual_result"].map(actual_map) != cls)])
        fn = len(df[(df["pred_winner"] != cls) & (df["actual_result"].map(actual_map) == cls)])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
