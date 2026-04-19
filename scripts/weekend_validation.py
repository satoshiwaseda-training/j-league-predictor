"""
scripts/weekend_validation.py

2026-04-18 の本命/最強層 7試合に各モデルを適用して、HIGH層正答率を比較する。
結果はスクリーンショットの予測値 + nikkan スコアから手動で入力。
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_improvement_test import (
    classify_prediction, label_baseline,
)
from label_threshold_sweep import (
    label_kb_floor, label_ka_w, label_ka_w_kb,
)


# 2026-04-18 本命・最強ラベル試合 (screenshot 由来)
# h,d,a は Gemini 補正後のカードに表示された %
# original_label はスクリーンショット上の戦略ラベル (draw_alert の再現に使う)
# h_form_w は screenshot の W/L/D 直近5戦から W 数をカウント
MATCHES = [
    {"name": "鹿島×浦和",  "h": 73, "d": 19, "a":  8, "h_form": "LLWLL", "actual": "home", "original_label": "本命"},
    {"name": "東京Ｖ×千葉","h": 52, "d": 26, "a": 22, "h_form": "DLLWW", "actual": "home", "original_label": "最強"},
    {"name": "仙台×栃木C", "h": 72, "d": 19, "a":  9, "h_form": "DWWLL", "actual": "home", "original_label": "本命"},
    {"name": "徳島×金沢",  "h": 94, "d":  5, "a":  1, "h_form": "WWWWW", "actual": "home", "original_label": "本命"},
    {"name": "湘南×群馬",  "h": 96, "d":  3, "a":  1, "h_form": "WDDLL", "actual": "away", "original_label": "本命"},
    {"name": "大宮×磐田",  "h": 81, "d": 13, "a":  6, "h_form": "DDWWL", "actual": "away", "original_label": "本命"},
    {"name": "甲府×藤枝",  "h": 57, "d": 27, "a": 16, "h_form": "WLWLL", "actual": "away", "original_label": "最強"},
]


def _c_from_label(m: dict) -> dict:
    """screenshot 上のラベルから (confidence, draw_alert) を逆算し、
    max_prob は post-Gemini の値を採用する。
    """
    label = m["original_label"]
    confidence = {"最強": "high", "本命": "high",
                  "波乱狙い": "medium", "組み合わせ": "medium",
                  "スキップ": "low", "見送り": "low"}[label]
    draw_alert = label in ("最強", "波乱狙い", "スキップ")
    return {
        "confidence": confidence,
        "draw_alert": draw_alert,
        "max_prob": max(m["h"], m["d"], m["a"]),
    }


def evaluate_strategy(name: str, label_fn):
    print(f"\n=== {name} ===")
    kept = []  # 最強 or 本命 に残った試合
    for m in MATCHES:
        h_form_w = sum(1 for x in m["h_form"] if x == "W")
        c = _c_from_label(m)
        label = label_fn(c, h_form_w)
        hit = "○" if m["actual"] == "home" else "✗"
        print(f"  {m['name']:<14} h={m['h']:>3}% draw_alert={str(c['draw_alert']):<5} "
              f"form_w={h_form_w} conf={c['confidence']:<6} → {label:<8} (結果={m['actual']} {hit})")
        if label in ("最強", "本命"):
            kept.append(m)
    hits = sum(1 for m in kept if m["actual"] == "home")
    if kept:
        print(f"  -- 高確信層: n={len(kept)} hit={hits} acc={hits/len(kept)*100:.1f}%")
    else:
        print(f"  -- 高確信層: n=0 (全試合が降格)")
    return len(kept), hits


if __name__ == "__main__":
    print("=" * 70)
    print("  2026-04-18 週末データへの各モデル適用")
    print("=" * 70)

    results = []
    results.append(("baseline", *evaluate_strategy("baseline (現行)",
                                                   lambda c, fw: label_baseline(c))))
    for w in [1, 2, 3]:
        results.append((f"改修A w<{w}",
                        *evaluate_strategy(f"改修A w<{w}",
                                           lambda c, fw, w=w: label_ka_w(c, fw, w))))
    for floor in [0.50, 0.60, 0.65]:
        results.append((f"改修B f={floor}",
                        *evaluate_strategy(f"改修B floor={floor}",
                                           lambda c, fw, fl=floor: label_kb_floor(c, fl))))
    for w in [2, 3]:
        for floor in [0.60, 0.65]:
            results.append((f"改修A+B w<{w} f={floor}",
                            *evaluate_strategy(f"改修A+B w<{w} floor={floor}",
                                               lambda c, fw, w=w, fl=floor: label_ka_w_kb(c, fw, w, fl))))

    print("\n" + "=" * 70)
    print("  週末HIGH層正答率サマリ")
    print("=" * 70)
    print(f"  {'model':<28} {'n':>3} {'hit':>4} {'acc':>8}")
    for name, n, hits in results:
        acc = hits/n*100 if n else 0
        print(f"  {name:<28} {n:>3} {hits:>4} {acc:>7.1f}%")
