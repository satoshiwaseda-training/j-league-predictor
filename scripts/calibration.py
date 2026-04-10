"""
scripts/calibration.py - 確率キャリブレーション

Temperature Scaling と Isotonic Regression を実装し、
既存モデルの出力確率を validation setで校正する。

使い方:
    # 1. トレーニング予測(probs, labels)を用意
    # 2. calibrator を fit
    # 3. 他の予測に apply
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any

LABELS = ["away", "draw", "home"]


# ────────────────────────────────────────────────────────
# Temperature Scaling
# ────────────────────────────────────────────────────────

def _log_loss(probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    """
    probs: (N, 3) float
    y_true_idx: (N,) int (0/1/2)
    """
    p = np.clip(probs, 1e-9, 1.0)
    # 各サンプルの真のクラスの確率
    row_idx = np.arange(len(y_true_idx))
    true_p = p[row_idx, y_true_idx]
    return float(-np.mean(np.log(true_p)))


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """
    probs (N, 3) → temperature scaling 適用後の probs
    logits = log(p) を T で割ってsoftmax
    """
    eps = 1e-9
    logits = np.log(np.clip(probs, eps, 1.0))
    scaled = logits / T
    # softmax (数値安定版)
    m = np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled - m)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fit_temperature(probs_train: np.ndarray, y_train_idx: np.ndarray,
                     T_range=(0.3, 3.0), n_grid: int = 60) -> float:
    """
    Temperature scaling のグリッドサーチで最適 T を見つける。
    """
    best_T = 1.0
    best_ll = float("inf")
    Ts = np.linspace(T_range[0], T_range[1], n_grid)
    for T in Ts:
        calib = _apply_temperature(probs_train, T)
        ll = _log_loss(calib, y_train_idx)
        if ll < best_ll:
            best_ll = ll
            best_T = float(T)
    return best_T


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0
        self.fitted = False

    def fit(self, probs_train: np.ndarray, y_train_idx: np.ndarray) -> "TemperatureScaler":
        self.T = fit_temperature(probs_train, y_train_idx)
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return probs
        return _apply_temperature(probs, self.T)


# ────────────────────────────────────────────────────────
# Isotonic Regression (class-wise)
# ────────────────────────────────────────────────────────

class IsotonicCalibrator:
    """
    クラスごとに IsotonicRegression を fit し、
    各クラスの確率を独立に校正してから再正規化する。
    """

    def __init__(self):
        self.iso = {}  # class_idx -> IsotonicRegression
        self.fitted = False

    def fit(self, probs_train: np.ndarray, y_train_idx: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        for c in range(3):
            y_binary = (y_train_idx == c).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs_train[:, c], y_binary)
            self.iso[c] = ir
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return probs
        out = np.zeros_like(probs)
        for c in range(3):
            out[:, c] = self.iso[c].transform(probs[:, c])
        # 再正規化
        row_sum = np.sum(out, axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
        return out / row_sum


# ────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────

def predictions_to_arrays(predictions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    predictions (backtest出力) から (probs, y_true_idx) を返す
    labels順: away=0, draw=1, home=2
    """
    probs = np.array([[p["prob_away"], p["prob_draw"], p["prob_home"]] for p in predictions])
    label_to_idx = {"away": 0, "draw": 1, "home": 2}
    y_idx = np.array([label_to_idx[p["actual"]] for p in predictions])
    return probs, y_idx


def compute_ece(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) - 多クラス版
    各ビンで (予測確率平均) と (実際の正答率) の差を重み付け平均
    """
    n = len(y_true_idx)
    if n == 0:
        return 0.0
    # argmaxの確率とそれが正解かどうか
    preds = np.argmax(probs, axis=1)
    confidences = probs[np.arange(n), preds]
    accuracies = (preds == y_true_idx).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_conf - bin_acc)
    return float(ece)
