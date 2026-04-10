# Hybrid v9 統合モデルレポート (hybrid_v9_integration.md)

> 作成日: 2026-04-10

---

## 統合モデルの構成

### モデル選択ロジック
```
入力: v7予測, Skellam dynamic予測
├── v7 draw警戒判定 (draw>=25% かつ |H-A|<10pp)
│   └── Yes → v7採用
├── Skellam高確信判定 (max>=50% かつ 非draw)
│   └── Yes → Skellam採用
└── else → v7とSkellamの等重み平均
```

### 動的draw_boost
Skellam内部で `compute_dynamic_draw_boost()` により per-match 計算:
- λ接近度 (lam_home vs lam_away)
- ELO接近度 (elo_h vs elo_a)
- xG接近度 (xg_home vs xg_away, あれば)
- 3シグナル平均からステップ関数で 0.0/0.04/0.08/0.12 のいずれかを選択

**固定値0.06は廃止、per-matchで変動。**

---

## 統合モデルの性能

### val=2025 (n=377)
| モデル | Acc | F1 | LogL | Brier | ECE | Draw# | DrawF1 |
|--------|-----|----|----|-------|-----|-------|--------|
| v7 primary | 0.478 | 0.421 | 1.053 | 0.634 | 0.061 | 44 | 0.227 |
| Skellam dyn | 0.456 | 0.416 | 1.059 | 0.636 | 0.057 | 116 | 0.319 |
| **hybrid_v9** | **0.480** | **0.432** | 1.056 | 0.634 | 0.065 | **54** | **0.265** |

### 2026 holdout (n=41)
| モデル | Acc | F1 | Draw# | DrawF1 |
|--------|-----|----|-------|--------|
| v7 primary | 0.342 | 0.258 | 1 | 0 |
| Skellam dyn | 0.439 | 0.392 | 12 | 0.364 |
| **hybrid_v9** | **0.366** | **0.317** | 2 | 0.167 |

---

## どの条件でどのモデルが選ばれたか

### val=2025 選択割合
| 選択 | 件数 | 割合 |
|------|------|------|
| v7 (draw警戒) | 177 | 46.9% |
| Skellam (高確信) | 101 | 26.8% |
| weighted | 99 | 26.3% |

### 2026 holdout 選択割合
| 選択 | 件数 | 割合 |
|------|------|------|
| v7 (draw警戒) | 16 | 39.0% |
| Skellam (高確信) | 9 | 22.0% |
| weighted | 16 | 39.0% |

→ 各モデルがバランスよく貢献している

---

## v7超えの有無

### val=2025
| 指標 | v7 | hybrid_v9 | 差分 | 判定 |
|------|-----|-----------|------|------|
| Accuracy | 0.478 | 0.480 | +0.2pp | わずかに超える |
| **F1 macro** | 0.421 | **0.432** | **+1.1pp** | **明確に超える** |
| LogL | 1.053 | 1.056 | +0.003 | わずかに悪化 |
| Brier | 0.634 | 0.634 | 0 | 同等 |
| Draw# | 44 | **54** | +10 | 制約40-55内 |
| **DrawF1** | 0.227 | **0.265** | **+3.8pp** | **改善** |

### 2026 holdout
| 指標 | v7 | hybrid_v9 | 差分 |
|------|-----|-----------|------|
| Accuracy | 0.342 | **0.366** | **+2.4pp** |
| F1 macro | 0.258 | **0.317** | **+5.9pp** |
| LogL | 1.099 | 1.086 | -0.013 |

**2026 holdoutでも全指標で v7 を上回る**

---

## 成功条件チェック

| 条件 | 結果 | 判定 |
|------|------|------|
| v7を上回る accuracy | val +0.2pp, 2026 +2.4pp | OK |
| v7を上回る F1 | val +1.1pp, 2026 +5.9pp | OK |
| draw数 40-55 | val=54件 | OK |
| 2026 holdoutで安定 | acc/F1ともに改善 | OK |

**全条件クリア**

---

## 実装状況

| ファイル | 追加内容 |
|---------|---------|
| scripts/skellam_model.py | `compute_dynamic_draw_boost()`, `predict_skellam_dynamic()` |
| scripts/predict_logic.py | `compute_hybrid_v9()` |
| scripts/model_comparison.py | `_run_skellam_dynamic_walk_forward()`, hybrid_v9評価ロジック |

### 本番UIへの影響
- **UIは一切変更なし**
- `compute_hybrid_v9()` は独立関数として追加 (呼び出し側が制御)
- 現状は v7 primary のまま

### 次のステップ候補
1. hybrid_v9 を shadow として保存 (v8.1 shadowと併走)
2. 2026 holdout n>=80 で再評価
3. hybrid_v9 の継続優位が確認されたら primary昇格検討
