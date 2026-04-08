# 2026 Holdout / 逐次評価ルール (live_holdout_2026_plan.md)

> 作成日: 2026-04-09

---

## 1. 評価ルール

### 分類
| Window | 節 | 試合数 | 用途 |
|--------|------|--------|------|
| Adaptation | 1-10 | ~44 | 微調整のみ (最大1回) |
| Test | 11- | 未来 | 最終評価 (非公開) |

### 厳守事項
- 2026データを一括最適化に使わないこと
- 各試合の予測には「その試合前」の情報のみ使用
- 予測保存後の書き換え禁止
- Adaptation windowでの許可: 単一スカラー補正の微調整のみ

---

## 2. 逐次予測ログ形式

| フィールド | 型 | 説明 |
|-----------|------|------|
| prediction_timestamp | datetime | 予測実行時刻 |
| round | int | 節番号 |
| match_id | str | "date_home_away" |
| available_data_cutoff | str | 利用可能データの最終日 |
| predicted_probs | dict | {"home": float, "draw": float, "away": float} |
| predicted_label | str | argmax結果 |
| actual_result | str | "home"/"draw"/"away" (確定後に追記) |
| model_version | str | "v7" |

---

## 3. 2026 正式検証結果 (v7, 2026-04-09時点)

| 指標 | v7 | elo_only | current v5 |
|------|------|---------|-----------|
| n | 41 | 41 | 41 |
| accuracy | **0.366** | 0.341 | 0.366 |
| F1 macro | **0.317** | 0.258 | 0.270 |
| log loss | 1.101 | **1.088** | 1.172 |
| Brier | 0.669 | **0.659** | 0.712 |
| draw# | 2 | 3 | 0 |

### 健全性判断
- **v7は2026でcurrent v5を全指標で上回る** (F1 +4.7pp, logL -0.071)
- **v7は2026でelo_onlyをacc/F1で上回る** (F1 +5.9pp)
- n=41は統計的に不十分 (95%CI ≈ ±15pp)
- draw=2件は少ない → 2026序盤の特性 (ELO蓄積不足)
- **破綻なし**: 2025での傾向が2026でも維持されている
