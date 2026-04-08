# 実験サマリ (experiment_summary.md)

> 実施日: 2026-04-08
> ブランチ: improve/backtest-framework

---

## 1. ベースライン性能

| 指標 | train (n=20) | val (n=9) | all (n=29) |
|------|-------------|-----------|------------|
| accuracy | 0.4000 | 0.2222 | 0.3448 |
| macro F1 | 0.2843 | 0.1786 | 0.2548 |
| log loss | 1.2394 | 1.2235 | 1.2345 |
| Brier score | 0.7419 | 0.7810 | 0.7540 |

**重大な問題**: ベースラインモデルは**drawを一度も予測しない**。
混同行列のdraw列が全て0 → 3クラス分類として機能していない。

ベースライン重み (v5の有効6パラメータを正規化):
```python
{
    "team_strength": 0.2044,
    "attack_rate": 0.1363,
    "defense_rate": 0.1022,
    "recent_form": 0.2559,
    "home_advantage": 0.1363,
    "capital_power": 0.1704,
}
```

---

## 2. 最良モデル性能 (Optuna 23 trials)

| 指標 | train (n=20) | val (n=9) | 改善幅(val) |
|------|-------------|-----------|------------|
| accuracy | 0.4500 | 0.3333 | **+11.1pp** |
| macro F1 | 0.3155 | 0.2593 | **+8.1pp** |
| log loss | 1.1303 | 1.1658 | **-0.058** |
| Brier score | 0.6837 | 0.7259 | **-0.055** |

最良重み:
```python
{
    "team_strength": 0.0101,   # 大幅減 (0.20→0.01)
    "attack_rate": 0.2133,     # 増 (0.14→0.21)
    "defense_rate": 0.0744,
    "recent_form": 0.2963,    # 微増 (0.26→0.30)
    "home_advantage": 0.1319,
    "capital_power": 0.0763,   # 減 (0.17→0.08)
    "elo": 0.1977,            # 【新規】ELO重み20%
}
```

最良ハイパーパラメータ:
```python
{
    "sigmoid_k": 5.76,         # 元3.0 → より急峻なシグモイド
    "sigmoid_m": 0.207,        # 元0.30 → シフト幅縮小
    "base_home": 0.301,        # 元0.40 → ホームバイアス大幅減
    "base_draw": 0.260,        # 元0.25 → draw微増
    "base_away": 0.440,        # 元0.35 → アウェイバイアス増
    "form_n": 5,               # 変化なし
    "draw_closeness_boost": 0.064, # 【新規】接近時draw+6.4%
    "elo_k": 28.1,             # 【新規】ELO K-factor
    "elo_home_bonus": 85.6,    # 【新規】ELOホームボーナス
}
```

---

## 3. 改善に効いた要因

### 3.1 最大効果: ベース確率の修正
- home 40%→30%, away 35%→44% への修正が最大効果
- J1 2026序盤はアウェイ勝率が高い (16/44=36% vs home 18/44=41%)
- **注意**: これは2026序盤のデータに過適合している可能性が高い

### 3.2 ELOレーティング導入 (重み20%)
- team_strength (勝点ベース) の代替として有効
- 試合ごとに動的更新されるため、シーズン序盤でも差が出る
- K=28, home_bonus=86 は文献値 (K=20-40) と整合

### 3.3 draw_closeness_boost
- 実力接近時にdraw確率を+6.4%するブースト
- drawを予測できるようになった (ベースラインでは0件)

### 3.4 team_strength → 0.01 に低下
- 勝点ベースの強度はELOに置換された
- シーズン序盤は勝点が小さく不安定→ELOの方が安定

---

## 4. 過学習リスク評価

| リスク要因 | 評価 | 詳細 |
|-----------|------|------|
| サンプル数 | **極高** | val=9件は統計的に無意味に近い (1件≈11%変動) |
| パラメータ数 | **高** | 12個のパラメータを9件のvalで最適化 |
| train-val gap | 中 | train 45% vs val 33% (12pp差) |
| base_away増加 | **高** | 2026序盤のアウェイ優勢に特化した可能性 |

**結論: 過学習リスクは極めて高い。本番反映は現時点では不推奨。**

---

## 5. 本番反映可否

### 反映可能 (安全):
1. **死にパラメータ6個の除去** → 15.3%の無駄な重みを排除
2. **EloSystemクラスの追加** → predict_logic.pyに新機能として追加可能
3. **バックテスト基盤** → scripts/backtest_runner.py として独立

### 反映不推奨 (要追加検証):
1. **重みの変更** → 44試合の過学習リスク
2. **base確率の変更** → シーズン全体のデータで検証必要
3. **draw_closeness_boost** → 効果はあるが汎化性未確認

---

## 6. 次の改善候補

### 優先度高
1. **データ量の増加**: 2024-2025シーズンのデータ取得 (get_past_resultsの年指定対応)
   - 2シーズン分 (~600試合) あれば統計的に有意な評価が可能
   - data_fetcherにyearパラメータ追加が必要

2. **holdoutテスト**: 2026 第5節以降をholdoutとして確保

3. **drawモデルの根本改善**: base_drawの動的化 (例: 両チームの得失点差が近いほどdraw確率上昇)

### 優先度中
4. **Dixon-Colesモデル**: ポアソン分布ベースの勝敗モデル導入
5. **シーズン序盤のシュリンクage強化**: 試合数が少ないときのprior設計
6. **クロスバリデーション**: Leave-one-section-out (節単位)

### 優先度低
7. **xG特徴量のバックテスト統合** (J1のみ)
8. **ロジスティック回帰への移行** (十分なデータ蓄積後)

---

## 7. 構築された実験基盤

| ファイル | 役割 |
|---------|------|
| `scripts/backtest_runner.py` | バックテスト実行・Optuna最適化・ログ保存 |
| `experiment_logs.csv` | 全実験の結果ログ (CSV) |
| `backtest_results/` | 詳細結果JSON |
| `docs/model_audit.md` | 現状分析レポート |
| `docs/program.md` | 実験計画書 |
| `docs/experiment_summary.md` | 本ファイル (実験結果サマリ) |

### 使い方
```bash
# ベースライン評価のみ
python scripts/backtest_runner.py

# Optuna最適化
python scripts/backtest_runner.py --optimize --n-trials 200 --patience 50

# カスタム実験
python scripts/backtest_runner.py --experiment-id my_exp --notes "description"
```
