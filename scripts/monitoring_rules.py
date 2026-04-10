"""
scripts/monitoring_rules.py - Primary モデル監視ルール定義

hybrid_v9.1 の自動降格条件・早期警戒ルール・予測ストアスキーマを定義する。
ロジック本体は変更しない。
"""

from __future__ import annotations


# ────────────────────────────────────────────────────────
# スキーマバージョン
# ────────────────────────────────────────────────────────

PREDICTION_SCHEMA_VERSION = "v2"  # 2026-04-11: primary/baseline/shadow 3層構造

PREDICTION_SCHEMA_FIELDS = [
    "id", "_key", "saved_at", "division", "match",
    "model_version", "baseline_model_version", "role",
    "schema_version",
    "prediction",           # primary (hybrid_v9.1)
    "baseline_prediction",  # v7 refined
    "shadow_prediction",    # v8.1
    "actual",
]

PREDICTION_SUBFIELDS = {
    "prediction": [
        "home_win_prob", "draw_prob", "away_win_prob",
        "predicted_score", "confidence", "pred_winner",
        "model", "hybrid_selection",
    ],
    "baseline_prediction": [
        "home_win_prob", "draw_prob", "away_win_prob",
        "pred_winner", "model_version",
    ],
    "shadow_prediction": [
        "home_win_prob", "draw_prob", "away_win_prob",
        "pred_winner", "model_version",
    ],
    "actual": ["score", "winner"],
}


# ────────────────────────────────────────────────────────
# 監視ウィンドウ
# ────────────────────────────────────────────────────────

ROLLING_WINDOWS = {
    "short":  20,  # 直近20試合: 早期警戒
    "medium": 40,  # 直近40試合: 中期トレンド
    "formal": 80,  # 80試合: 正式定着判定
}


# ────────────────────────────────────────────────────────
# 自動降格しきい値 (primary vs baseline)
# ────────────────────────────────────────────────────────

# 各指標でbaselineがprimaryを上回る幅 (負値=primary優位)
DOWNGRADE_THRESHOLDS = {
    "accuracy_gap":  -0.05,   # baseline - primary >= 5pp → 警戒
    "f1_gap":        -0.05,   # baseline - primary >= 5pp → 警戒
    "log_loss_gap":   0.05,   # primary - baseline >= 0.05 → 警戒 (logLは低い方が良い)
}

# 警戒レベル
WARNING_LEVELS = {
    "GREEN":  "正常 (primary優位または同等)",
    "YELLOW": "注意 (1指標でbaseline優位)",
    "ORANGE": "警告 (2指標でbaseline優位)",
    "RED":    "危険 (3指標でbaseline優位 → 降格推奨)",
}

# 最小サンプル数
MIN_SAMPLES_FOR_WARNING = 20   # 20試合未満では警戒判定しない
MIN_SAMPLES_FOR_DOWNGRADE = 40  # 40試合未満では自動降格しない


# ────────────────────────────────────────────────────────
# 判定関数
# ────────────────────────────────────────────────────────

def evaluate_warning_level(
    primary_metrics: dict,
    baseline_metrics: dict,
    n: int,
) -> tuple[str, list[str]]:
    """
    primary vs baseline の指標比較から警戒レベルを決定する。

    Returns
    -------
    (level, reasons): ("GREEN"/"YELLOW"/"ORANGE"/"RED", 理由リスト)
    """
    if n < MIN_SAMPLES_FOR_WARNING:
        return "GREEN", [f"サンプル不足 (n={n} < {MIN_SAMPLES_FOR_WARNING})"]

    p_acc = primary_metrics.get("accuracy", 0)
    b_acc = baseline_metrics.get("accuracy", 0)
    p_f1 = primary_metrics.get("f1_macro", 0)
    b_f1 = baseline_metrics.get("f1_macro", 0)
    p_ll = primary_metrics.get("log_loss", 0)
    b_ll = baseline_metrics.get("log_loss", 0)

    alerts = []

    # acc判定 (primary - baseline)
    acc_gap = p_acc - b_acc
    if acc_gap <= DOWNGRADE_THRESHOLDS["accuracy_gap"]:
        alerts.append(f"accuracy劣化 (primary {p_acc:.3f} vs baseline {b_acc:.3f}, gap={acc_gap:+.3f})")

    # F1判定
    f1_gap = p_f1 - b_f1
    if f1_gap <= DOWNGRADE_THRESHOLDS["f1_gap"]:
        alerts.append(f"F1劣化 (primary {p_f1:.3f} vs baseline {b_f1:.3f}, gap={f1_gap:+.3f})")

    # log_loss判定
    ll_gap = p_ll - b_ll
    if ll_gap >= DOWNGRADE_THRESHOLDS["log_loss_gap"]:
        alerts.append(f"log_loss劣化 (primary {p_ll:.3f} vs baseline {b_ll:.3f}, gap={ll_gap:+.3f})")

    n_alerts = len(alerts)
    if n_alerts == 0:
        return "GREEN", ["全指標でprimary優位または同等"]
    elif n_alerts == 1:
        return "YELLOW", alerts
    elif n_alerts == 2:
        return "ORANGE", alerts
    else:
        return "RED", alerts


def should_auto_downgrade(
    level: str,
    n: int,
    consecutive_red: int = 0,
) -> tuple[bool, str]:
    """
    自動降格判断。

    降格条件:
    - RED警戒がn>=40で発生
    - または RED警戒が連続3回以上続いた場合

    Returns
    -------
    (should_downgrade, reason)
    """
    if n < MIN_SAMPLES_FOR_DOWNGRADE:
        return False, f"サンプル不足 (n={n} < {MIN_SAMPLES_FOR_DOWNGRADE})"
    if level == "RED" and n >= MIN_SAMPLES_FOR_DOWNGRADE:
        return True, f"RED警戒 (n={n}, consecutive_red={consecutive_red}) → 降格推奨"
    if consecutive_red >= 3:
        return True, f"RED警戒が{consecutive_red}回連続 → 降格推奨"
    return False, f"降格不要 (level={level}, n={n})"
