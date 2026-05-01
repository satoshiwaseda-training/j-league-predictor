"""
prediction_store.py - 予測履歴の保存・管理
data/predictions.json に永続化 (同一試合は上書き)
data/predictions.log.jsonl に追記専用ログ (全予測イベントをタイムスタンプ順に保持)
"""
from __future__ import annotations

import csv
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

STORE_PATH = Path(__file__).parent / "data" / "predictions.json"
# 追記専用ログ: 同一試合を複数回予測しても全て残す
LOG_PATH = Path(__file__).parent / "data" / "predictions.log.jsonl"
# 分析・精度改善用の自動アーカイブ: J1/J2と試合日で整理し、1予測につき1 JSONを保存
# 既定はローカル Windows パス。Streamlit Cloud (Linux) や別 PC では環境変数
# JLEAGUE_ARCHIVE_DIR で上書きできる。指定パスへの書き込みに失敗しても
# _archive_prediction_event 内の try/except により予測本体は影響を受けない。
_DEFAULT_ARCHIVE_DIR = Path(r"C:\Users\User\OneDrive\Desktop\Jリーグ予測結果")
_ARCHIVE_DIR_ENV = os.getenv("JLEAGUE_ARCHIVE_DIR", "").strip()
ARCHIVE_DIR = Path(_ARCHIVE_DIR_ENV) if _ARCHIVE_DIR_ENV else _DEFAULT_ARCHIVE_DIR
ARCHIVE_INDEX_PATH = ARCHIVE_DIR / "index.csv"
ARCHIVE_ERROR_LOG_PATH = Path(__file__).parent / "data" / "prediction_archive_errors.log"

# Cloud (Linux/Mac で env 未設定) では archive をスキップする。
# Windows パスを Linux で mkdir すると、cwd に "C:\Users\..." 名のフォルダが
# できてしまう副作用を防ぐ。env で明示指定された場合は OS 問わず有効。
_ARCHIVE_ENABLED = bool(_ARCHIVE_DIR_ENV) or (os.name == "nt")


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
    adjustments: dict | None = None,
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
    adjustments : fan/travel 補正の監視情報 (optional)
        {fan_applied, travel_applied, fan_value, travel_value,
         pre_h, pre_d, pre_a, post_h, post_d, post_a, argmax_changed}
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
    entry = {
        "id":       pred_id,
        "_key":     key,
        "saved_at": datetime.now().isoformat(),
        "schema_version": "v2",
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
        "adjustments": adjustments,
        "baseline_prediction": baseline_entry,
        "shadow_prediction":   shadow_entry,
        "actual": None,
    }
    predictions.insert(0, entry)
    _write_all(predictions)
    _append_log(entry)
    _archive_prediction_event(entry)
    return pred_id


def _safe_filename_part(value: object, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text[:60] or fallback


def _archive_division_name(value: object) -> str:
    division = str(value or "unknown").strip().upper()
    return _safe_filename_part(division, "UNKNOWN")


def _archive_match_date(match: dict) -> str:
    raw_date = str(match.get("date") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
        return raw_date
    return datetime.now().strftime("%Y-%m-%d")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).parent))
    except ValueError:
        return str(path)


def _prediction_label(pred_winner: str, match: dict) -> str:
    if pred_winner == "home":
        return f"ホーム勝利: {match.get('home', '')}"
    if pred_winner == "away":
        return f"アウェー勝利: {match.get('away', '')}"
    if pred_winner == "draw":
        return "引き分け"
    return str(pred_winner or "")


def _build_prediction_result(entry: dict, match: dict, prediction: dict) -> dict:
    pred_winner = prediction.get("pred_winner", "")
    return {
        "home_win_prob": prediction.get("home_win_prob", ""),
        "draw_prob": prediction.get("draw_prob", ""),
        "away_win_prob": prediction.get("away_win_prob", ""),
        "predicted_score": prediction.get("predicted_score", ""),
        "pred_winner": pred_winner,
        "pred_winner_label": _prediction_label(pred_winner, match),
        "confidence": prediction.get("confidence", ""),
        "model": prediction.get("model", ""),
        "model_version": entry.get("model_version", ""),
        "baseline_prediction": entry.get("baseline_prediction"),
        "shadow_prediction": entry.get("shadow_prediction"),
    }


def _build_human_summary(entry: dict, match: dict, prediction: dict) -> dict:
    pred_winner = prediction.get("pred_winner", "")
    return {
        "リーグ": _archive_division_name(entry.get("division")),
        "試合日": _archive_match_date(match),
        "時刻": match.get("time", ""),
        "ホーム": match.get("home", ""),
        "アウェー": match.get("away", ""),
        "会場": match.get("venue", ""),
        "予想勝敗": _prediction_label(pred_winner, match),
        "予想勝敗コード": pred_winner,
        "予想スコア": prediction.get("predicted_score", ""),
        "自信度": prediction.get("confidence", ""),
        "ホーム勝率": prediction.get("home_win_prob", ""),
        "引き分け率": prediction.get("draw_prob", ""),
        "アウェー勝率": prediction.get("away_win_prob", ""),
        "モデル": prediction.get("model", ""),
        "モデルバージョン": entry.get("model_version", ""),
        "保存ID": entry.get("id", ""),
        "保存日時": entry.get("saved_at", ""),
    }


def _archive_prediction_event(entry: dict) -> None:
    """Save each prediction event as J1/J2/date JSON plus CSV indexes.

    The archive is intentionally append-only so repeated predictions for the
    same match remain available for later accuracy analysis.
    """
    if not _ARCHIVE_ENABLED:
        # Cloud (Linux/Mac で JLEAGUE_ARCHIVE_DIR 未設定) ではスキップ
        return
    try:
        saved_at = str(entry.get("saved_at") or datetime.now().isoformat())
        try:
            saved_dt = datetime.fromisoformat(saved_at)
        except ValueError:
            saved_dt = datetime.now()

        match = entry.get("match", {}) or {}
        prediction = entry.get("prediction", {}) or {}
        human_summary = _build_human_summary(entry, match, prediction)
        prediction_result = _build_prediction_result(entry, match, prediction)
        division = _archive_division_name(entry.get("division"))
        match_date = _archive_match_date(match)
        day_dir = ARCHIVE_DIR / division / match_date
        day_dir.mkdir(parents=True, exist_ok=True)

        timestamp = saved_dt.strftime("%Y%m%d_%H%M%S_%f")
        home = _safe_filename_part(match.get("home"), "home")
        away = _safe_filename_part(match.get("away"), "away")
        pred_id = _safe_filename_part(entry.get("id"), "prediction")
        archive_path = day_dir / f"{timestamp}_{home}_vs_{away}_{pred_id}.json"

        archive_payload = {
            "event": "prediction_created",
            "archived_at": datetime.now().isoformat(),
            "archive_schema_version": "v2",
            "organization": {
                "division": division,
                "match_date": match_date,
                "folder": _display_path(day_dir),
            },
            "ai_result_judgement": {
                "status": "pending_actual_result",
                "actual_score": None,
                "actual_winner": None,
                "prediction_correct": None,
                "notes": "実際の試合結果が入ったら actual_score と actual_winner を照合する",
            },
            "match": match,
            "prediction_result": prediction_result,
            "予想結果": human_summary,
            "entry": entry,
        }
        archive_path.write_text(
            json.dumps(archive_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        index_row = {
            "リーグ": division,
            "試合日": match_date,
            "時刻": match.get("time", ""),
            "ホーム": match.get("home", ""),
            "アウェー": match.get("away", ""),
            "予想勝敗": human_summary["予想勝敗"],
            "予想スコア": prediction.get("predicted_score", ""),
            "自信度": prediction.get("confidence", ""),
            "ホーム勝率": prediction.get("home_win_prob", ""),
            "引き分け率": prediction.get("draw_prob", ""),
            "アウェー勝率": prediction.get("away_win_prob", ""),
            "保存ID": entry.get("id", ""),
            "saved_at": saved_at,
            "id": entry.get("id", ""),
            "division": division,
            "match_date": match_date,
            "home": match.get("home", ""),
            "away": match.get("away", ""),
            "pred_winner": prediction.get("pred_winner", ""),
            "predicted_score": prediction.get("predicted_score", ""),
            "home_win_prob": prediction.get("home_win_prob", ""),
            "draw_prob": prediction.get("draw_prob", ""),
            "away_win_prob": prediction.get("away_win_prob", ""),
            "confidence": prediction.get("confidence", ""),
            "model_version": entry.get("model_version", ""),
            "model": prediction.get("model", ""),
            "actual_score": "",
            "actual_winner": "",
            "prediction_correct": "",
            "archive_path": _display_path(archive_path),
        }

        fieldnames = [
            "リーグ", "試合日", "時刻", "ホーム", "アウェー",
            "予想勝敗", "予想スコア", "自信度",
            "ホーム勝率", "引き分け率", "アウェー勝率", "保存ID",
            "saved_at", "id", "division", "match_date", "home", "away",
            "pred_winner", "predicted_score", "home_win_prob",
            "draw_prob", "away_win_prob", "confidence", "model_version",
            "model", "actual_score", "actual_winner", "prediction_correct",
            "archive_path",
        ]
        _append_archive_index(ARCHIVE_INDEX_PATH, fieldnames, index_row)
        _append_archive_index(day_dir / "index.csv", fieldnames, index_row)
    except Exception as exc:
        _append_archive_error(entry, exc)


def _append_archive_index(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig の空ファイルは BOM だけで3バイトになるため、ヘッダーなし扱いにする。
    should_write_header = not path.exists() or path.stat().st_size <= 3
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_archive_error(entry: dict, exc: Exception) -> None:
    try:
        ARCHIVE_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        error_entry = {
            "event": "prediction_archive_failed",
            "failed_at": datetime.now().isoformat(),
            "archive_dir": str(ARCHIVE_DIR),
            "id": entry.get("id"),
            "_key": entry.get("_key"),
            "division": entry.get("division"),
            "match": entry.get("match"),
            "error": str(exc),
        }
        with open(ARCHIVE_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_archive_root() -> Path:
    return ARCHIVE_DIR


def rebuild_prediction_archive_from_store() -> int:
    """既存の保存済み予測を指定フォルダへ再出力する。"""
    if not _ARCHIVE_ENABLED:
        return 0
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_INDEX_PATH.exists():
        ARCHIVE_INDEX_PATH.unlink()
    archived = 0
    for entry in reversed(load_all()):
        _archive_prediction_event(entry)
        archived += 1
    return archived


def _append_log(entry: dict) -> None:
    """追記専用ログに 1 行 JSON を append。
    data/predictions.json が上書きで消える既存試合の予測も、
    このログには全予測イベントがタイムスタンプ順に残る。
    書き込み失敗してもメイン保存は継続する (best-effort)。
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # ログ書き込みはメイン保存を邪魔しない
        pass


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
            # 結果反映イベントも JSONL ログに追記
            _append_log({
                "event":   "actual_updated",
                "id":      pred_id,
                "_key":    p.get("_key"),
                "updated_at": datetime.now().isoformat(),
                "actual":  p["actual"],
            })
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
