# 次世代モデル設計候補 (next_generation_model_plan.md)

> 作成日: 2026-04-09
> ステータス: 設計のみ (実装は将来ループ)

---

## 候補1: Dixon-Colesポアソンモデル

### 概要
各チームの攻撃力(alpha_i)と守備力(beta_i)をポアソン分布でモデル化。
home得点 ~ Poisson(alpha_home * beta_away * gamma)
away得点 ~ Poisson(alpha_away * beta_home)
gamma = ホームアドバンテージ係数

### 利点
- スコア予測が自然に出る (確率的にH-Aスコア���生成)
- draw確率が「得点数が同じ」確率として自然に算出
- 学術的に確���された手法 (Dixon & Coles 1997)
- パラメータが少なく過学習リスク低い

### 課題
- フォーム・資本力・ELO等の追加特徴量の組み込みが複雑
- 760試合のデータでパラメータ推定が安定するか要検証
- 実装コス��: 最尤推定のイテレーション必要

### 実装見積
- scipy.optimizeでのMLE実装: 中程度
- 既存の3ロジットとの併用可能: 高い

---

## 候補2: closeness非線形変換

### 概要
現在: closeness = max(0, 1 - |raw_adv| * 3)  (線形)
提案: closeness = exp(-raw_adv^2 / sigma^2)  (ガウシアン)

### 利点
- 実力差が大きい試合でclosenessが0に急速に落ちる
- draw logitの制御がよりスムーズ
- sigma をデータから推定可能

### 課題
- 現状のclosenessで十分機能している (val F1=0.435)
- 改善幅は小さい可能性

---

## 候補3: ELO拡張

### 3a. Goal-weighted ELO
勝敗だけでなくゴール差でK-factorを変動:
K_eff = K * (1 + 0.5 * |goal_diff|) for goal_diff > 1

### 3b. Seasonal reset
シーズン開始時にELOを平均方向に20%回帰:
elo_new = elo_old * 0.8 + 1500 * 0.2

### 3c. Glicko-2
不確実性(RD)を追跡するGlicko-2への発展。
長期離脱チ���ムの不確実性を自動管理。

---

## 候補4: Gemini強化

### 4a. Few-shot examples
プロンプトに過去の正答例(3-5件)を追加し、
Geminiの出力形式と確率感度を誘��。

### 4b. Self-consistency
同一試合に対してGeminiを3回実行し、多数決で最終予測。
コスト3倍だが安定性向上。

### 4c. Chain-of-thought制御
推論過程を構造化し、各パラメータへの言及を必須化。

---

## 優先度

| 候補 | 期待効果 | 実装コスト | 推奨時期 |
|------|---------|-----------|---------|
| 2. closeness非線形化 | 小 | 低 | 次ループ |
| 3b. Seasonal reset | 中 | 低 | 次ループ |
| 4a. Few-shot examples | 中 | 低 | 次ループ |
| 3a. Goal-weighted ELO | 中 | 中 | データ蓄積後 |
| 1. Dixon-Coles | 高 | 高 | 2026後半 |
| 3c. Glicko-2 | 中 | 高 | 将来 |
| 4b/4c. Gemini強化 | 不明 | 中 | 効果検証後 |

---

## 次の改善フェーズへの条件

以下が揃ったとき、次のモデル改善フェーズに移行:
1. 2026 holdout n >= 100 (��計的信頼性の確保)
2. v7 refinedの2026精度が安定 (直近5節で大きな変動なし)
3. Gemini統合の効果が定性的に確認された
4. 具体的な改善��説がデータから導出された
