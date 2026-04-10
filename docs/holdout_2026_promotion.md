# Primary Model 昇格レポート - hybrid_v9.1

> 作成日: 2026-04-11
> 昇格: hybrid_v9.1 → Primary (v7は baseline として保持)

---

## 昇格判断の根拠

### val=2025 (n=377)
| モデル | Acc | F1 | LogL | Draw# | DrawF1 |
|--------|-----|-----|------|-------|--------|
| v7 (旧primary) | 0.478 | 0.421 | 1.053 | 44 | 0.227 |
| **hybrid_v9.1 (新primary)** | 0.475 | **0.428** | 1.056 | **55** | **0.276** |

### 2026 Holdout (n=41)
| モデル | Acc | F1 | LogL | Brier | ECE | Draw# |
|--------|-----|-----|------|-------|-----|-------|
| **hybrid_v9.1** | **0.415** | **0.349** | **1.078** | **0.650** | **0.083** | 2 |
| v7 baseline | 0.342 | 0.258 | 1.099 | 0.667 | 0.113 | 1 |
| v8.1 shadow | 0.342 | 0.258 | 1.105 | 0.672 | 0.125 | 1 |

**2026 holdout で hybrid_v9.1 が全指標で勝利:**
- Accuracy: +7.3pp
- F1 macro: +9.1pp
- Log Loss: -0.021
- Brier: -0.017
- ECE: -0.030

---

## 実装構成

### Primary: hybrid_v9.1
- 場所: `scripts/predict_logic.py: compute_hybrid_v9()`
- 構成要素: v7 + Skellam dynamic の選択統合
- 選択ロジック:
  - v7 draw警戒 → v7採用
  - Skellam高確信 (max>=48% 非draw) → Skellam採用
  - Clear favorite (\|λ差\|>=0.5 or \|ELO差\|>=0.15) → Skellam採用
  - それ以外 → v7/Skellam平均

### Baseline: v7 refined (fallback)
- 場所: `scripts/predict_logic.py` の既存 MODEL_WEIGHTS/_3LOGIT_PARAMS
- 役割: Gemini reasoning生成、hybrid計算の入力、フォールバック

### Shadow: v8.1
- 場所: `scripts/predict_logic.py: compute_shadow_v8_1()`
- 役割: 内部比較用ログ

### Fallback 切り替え方法
```python
# scripts/predict_logic.py
PRIMARY_MODEL_VERSION = "hybrid_v9.1"  # or "v7_refined" で v7 に戻す
```
この定数を書き換えるだけでfallback可能。

---

## prediction_store スキーマ変更

各予測エントリに以下のフィールドが追加:
```json
{
  "model_version": "hybrid_v9.1",
  "baseline_model_version": "v7_refined",
  "prediction": {
    "home_win_prob": ...,
    "draw_prob": ...,
    "away_win_prob": ...,
    "hybrid_selection": "v7" | "skellam" | "weighted",
    ...
  },
  "baseline_prediction": {
    "home_win_prob": ...,
    "draw_prob": ...,
    "away_win_prob": ...,
    "pred_winner": ...,
    "model_version": "v7_refined"
  },
  "shadow_prediction": {
    ... (v8.1)
  }
}
```

---

## 選択割合

### 2026 holdout (n=41)
| 選択 | 件数 | 割合 |
|------|------|------|
| v7 (draw警戒) | 15 | 36.6% |
| Skellam (高確信/clear favorite) | 18 | 43.9% |
| weighted (0.5/0.5平均) | 8 | 19.5% |

---

## 継続監視

- 現状 n=41 は統計的信頼性不十分
- **n>=80 到達時に正式定着判定**
- 判定スクリプト: `python scripts/shadow_comparison.py`
- 結果: `backtest_results/holdout_2026_comparison.json`

### 判定基準 (n>=80到達時)
- hybrid_v9.1 の accuracy > v7
- hybrid_v9.1 の F1 > v7
- hybrid_v9.1 の log loss <= v7
- 上記すべて満たせば正式定着
- 劣化が確認されれば `PRIMARY_MODEL_VERSION = "v7_refined"` に戻す
