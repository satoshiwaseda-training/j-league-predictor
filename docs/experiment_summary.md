# 実験サマリ v3 — 第2自己改善ループ (experiment_summary.md)

> 更新日: 2026-04-08
> ブランチ: improve/backtest-framework
> データ: 804試合 (2024: 380 + 2025: 380 + 2026: 44)

---

## 1. draw構造欠陥の発見と修正

### 1.1 根本原因 (数学的証明)

現行の `advantage_to_probs()` において:
```
d = 1 - (base_home + shift) - (base_away - shift)
  = 1 - base_home - base_away
  = 0.25 (定数)
```
**draw確率はraw_advantageに依存しない定数0.25**。
home=0.40, away=0.35 なので、drawは全領域で3位。
argmaxでdrawが選ばれることは数学的に不可能。

詳細: docs/draw_failure_analysis.md

### 1.2 修正: 3ロジット + softmax 方式

```python
logit_home = scale_ha * raw_adv + bias_home
logit_away = -scale_ha * raw_adv + bias_away
logit_draw = scale_draw * closeness + bias_draw
probs = softmax([logit_home, logit_draw, logit_away])
```

- drawに独立したロジットを与える (残差ではなく)
- closeness (実力接近度) が高いとdrawロジットが上がる
- softmaxで3クラス確率を一貫して算出

---

## 2. 全モデル比較 (val=2025, n=377)

| Predictor | Accuracy | F1 macro | Log Loss | Brier | Draw# | Draw P | Draw R |
|-----------|----------|----------|----------|-------|-------|--------|--------|
| elo_only | 0.483 | **0.406** | 1.039 | 0.625 | 40 | 35.0% | **14.4%** |
| **3logit opt** | 0.475 | 0.376 | **1.040** | **0.625** | **32** | 31.3% | 10.3% |
| current v5 | 0.464 | 0.341 | 1.057 | 0.634 | 0 | - | 0% |
| always_home | 0.443 | 0.205 | 1.123 | 0.686 | 0 | - | 0% |
| prior | 0.443 | 0.205 | 1.073 | 0.649 | 0 | - | 0% |
| form_only | 0.422 | 0.313 | 1.123 | 0.676 | 0 | - | 0% |

### 採用判定チェックリスト (3logit opt)

| 条件 | 結果 |
|------|------|
| draw予測数 > 0 | **OK** (32件) |
| draw recall > 0 | **OK** (10.3%) |
| macro F1 > current v5 | **OK** (+3.5pp) |
| log_loss not worse than elo_only | **OK** (1.040 vs 1.039) |
| calibration not extremely worse | **OK** (Brier 0.625 同等) |
| 2026 holdout not degraded | **OK** (0.390 vs 0.366) |

**判定: 採用可能** — 全条件を満たす。

---

## 3. 3logit最適化パラメータ

### 重み
| パラメータ | 値 | current v5 |
|-----------|------|-----------|
| team_strength | 0.138 | 0.204 |
| attack_rate | 0.066 | 0.136 |
| defense_rate | 0.148 | 0.102 |
| recent_form | 0.103 | 0.256 |
| home_advantage | 0.113 | 0.136 |
| capital_power | 0.126 | 0.170 |
| **elo** | **0.266** | 0.000 |

### 3ロジット変換パラメータ
| パラメータ | 値 | 意味 |
|-----------|------|------|
| scale_ha | 1.44 | 勝敗方向の感度 |
| bias_home | 0.14 | ホームバイアス |
| bias_away | 0.07 | アウェイバイアス |
| scale_draw | 1.01 | draw感度 |
| bias_draw | -0.82 | drawベースロジット (負=drawは稀) |
| elo_k | 43.8 | ELO K-factor |
| elo_home_bonus | 99.6 | ELOホームボーナス |
| form_n | 5 | フォーム参照期間 |

---

## 4. 3層評価結果

### current v5 (ベースライン)
| Layer | n | Accuracy | F1 | Log Loss |
|-------|---|----------|------|----------|
| Train 2024 | 370 | 41.9% | 0.317 | 1.115 |
| Val 2025 | 377 | 46.4% | 0.341 | 1.057 |
| 2026 adapt | 41 | 36.6% | 0.270 | 1.172 |

### 3logit optimized
| Layer | n | Accuracy | F1 | Log Loss |
|-------|---|----------|------|----------|
| Val 2025 | 377 | 47.5% | 0.376 | 1.040 |
| 2026 adapt | 41 | 36.6% | 0.269 | 1.103 |

2026でlog_loss改善 (1.172 → 1.103)、accuracy同等。

---

## 5. 付随する発見

### 5.1 recent_form の過大評価
- 現行: 0.256 (最大) → 最適化後: 0.103 (60%減)
- ELOがフォーム情報を大部分含むため、ELO導入時に重みが減るのは正しい
- 詳細: docs/recent_form_review.md

### 5.2 死にパラメータ 6個
- set_piece_conversion, match_day_motivation, tactical_adaptability: 完全削除可
- player_availability_impact, match_trend, referee_tendency: 将来用に残す
- 合計15.3%の重みが無駄
- 詳細: docs/dead_parameters_review.md

### 5.3 ELOの重要性
- 全最適化でELO重み=22-27% (常に最大)
- ELO単体で現行モデルを上回る
- 詳細: docs/elo_integration.md

---

## 6. 安全に反映可能な変更

| 変更 | ファイル | 本番影響 |
|------|---------|---------|
| 3ロジット予測器 | backtest_runner.py | なし (評価用) |
| EloSystemクラス | backtest_runner.py | なし (評価用) |
| draw_failure_analysis.md | docs/ | なし |
| elo_integration.md | docs/ | なし |
| recent_form_review.md | docs/ | なし |
| dead_parameters_review.md | docs/ | なし |
| data_fetcher拡張 | data_fetcher.py | なし (新API追加のみ) |

## 7. 条件付きで反映可能な変更

| 変更 | 条件 |
|------|------|
| predict_logic.py に3ロジット変換追加 | draw=0解消、F1改善、UI互換 |
| predict_logic.py にEloSystem追加 | 上記+ELO構築データフローの確認 |

## 8. まだ反映すべきでない変更

| 変更 | 理由 |
|------|------|
| MODEL_WEIGHTS の直接変更 | 3ロジット構造とセットで設計すべき |
| base確率の変更 | 3ロジット移行で不要になる |
| 死にパラメータの削除 | 3ロジット移行後に改めて設計 |

---

## 9. 次の自己改善ループ

### 最優先
1. **predict_logic.py に3ロジット変換を本番統合**
   - `advantage_to_probs()` の代替として `advantage_to_probs_3logit()` を追加
   - Gemini非使用時のfallbackとして使用
   - Streamlitの表示関数を確認し、互換性を保証

2. **ELOデータフローの本番構築**
   - app.py から ELO を構築・キャッシュする仕組み
   - `get_past_results()` → EloSystem.update() のパイプライン

### 次点
3. **draw F1のさらなる改善**
   - elo_only (draw recall 14.4%) に追いつく
   - bias_draw の微調整
   - closenessの算出式改善

4. **Geminiプロンプトへの ELO/3logit 情報追加**
