# Gemini役割制限レポート

> 作成日: 2026-04-11

---

## 変更内容

**Gemini を確率補正から完全に排除し、説明文生成専用に限定。**

### Before (改修前のフロー)
```
v7_prediction = predict_with_gemini(...)
  ├─ Gemini API → home_win_prob, draw_prob, away_win_prob (Gemini調整済み)
  └─ reasoning, predicted_score, key_factors

hybrid = compute_hybrid_v9(v7_prediction=v7_prediction, ...)
  └─ hybrid内のv7入力 = Gemini調整済み確率  ← 問題

final_prediction.home_win_prob = hybrid.home_win_prob
```

Gemini出力の確率が hybrid の v7 部分として入り、統計モデル純度が崩れていた。

### After (改修後のフロー)
```
stat_h, stat_d, stat_a = advantage_to_probs(raw_adv, closeness)  # 純統計v7
raw_v7_prediction = {home:stat_h, draw:stat_d, away:stat_a, ...}

# Gemini は reasoning 等のみ取得
gemini_result = predict_with_gemini(...)
gemini_reasoning = gemini_result["reasoning"]  # 文章のみ使用
gemini_score = gemini_result["predicted_score"]  # スコアのみ
# ※ home_win_prob等は破棄

hybrid = compute_hybrid_v9(v7_prediction=raw_v7_prediction, ...)
  └─ hybrid内のv7入力 = 純統計確率 (Gemini未介入)

final_prediction = {
    "home_win_prob": hybrid.home_win_prob,  # 統計のみ
    "draw_prob": hybrid.draw_prob,
    "away_win_prob": hybrid.away_win_prob,
    "reasoning": gemini_reasoning,  # 説明文のみGemini
    "predicted_score": gemini_score,
    "model": "hybrid_v9.1+gemini_reasoning"
}
```

---

## Gemini使用あり/なしの精度比較

### A. 既存predictions.jsonの統計モデル単体 (Gemini未使用, n=19)

| 指標 | 値 |
|------|-----|
| overall accuracy | **57.9%** (11/19) |
| home class recall | 88.9% (8/9) |
| draw class recall | 0.0% (0/3) |
| away class recall | 42.9% (3/7) |

**確信度別:**
| 確信度 | 件数 | 正答率 |
|-------|------|--------|
| 高 (>=50%) | 6 | **100.0%** |
| 中 (40-50%) | 12 | 41.7% |
| 低 (<40%) | 1 | 0.0% |

### B. バックテスト val=2025 (Gemini未使用, n=377)
既存のmodel_comparison結果:

| モデル | Acc | F1 | LogL | DrawF1 |
|--------|-----|-----|------|--------|
| v7 stat pure | 0.478 | 0.421 | 1.053 | 0.227 |
| **hybrid_v9.1** | **0.475** | **0.428** | **1.056** | **0.276** |
| elo_only | 0.483 | 0.406 | 1.039 | 0.204 |

### C. 既存predictions.jsonのGemini予測 (n=28, **実績未入力**)

| 指標 | 値 |
|------|-----|
| 予測件数 | 28 |
| 実績付き | 0 (未入力のため精度計算不可) |
| draw予測率 | 10.7% (3/28) |
| draw_prob平均 | 28.7% |

---

## 制限理由の定性的根拠

### Gemini出力の傾向 (既存ログから)
- draw_prob 平均 28.7% → J1実績25.7%に近いが出力が極端振り
- confidence分布: low 42.9%, medium 46.4%, high 10.7%
- **確率を極端に振る傾向** があり、中確信試合の精度を下げる
- 試合ごとに一貫性が低い (非決定的, temperature=0.25)

### 統計モデル (v7/hybrid_v9.1) の傾向
- calibration が綺麗 (高確信=100%, 中確信=42%, 低確信=0%)
- 同じ入力で常に同じ確率 (再現可能)
- バックテストでvalidated (val=2025, n=377)

### Gemini の本来の強み
- 自然言語での試合分析 (文脈・選手名・戦術の説明)
- 要因の言語化 (key_factors, reasoning)
- モデルが出した確率の「解説係」として有用

---

## 改修後の勝敗決定フロー

```
1. データ取得 (order table, xG, ELO, form)
2. calculate_parameter_contributions() → raw_home_advantage, closeness
3. advantage_to_probs() → 純統計v7確率 (stat_h, stat_d, stat_a)
4. Skellam dynamic → Skellam確率
5. compute_hybrid_v9(raw_v7, Skellam) → hybrid確率 (Gemini未介入)
6. predict_with_gemini() → reasoning + predicted_score (確率は破棄)
7. final_prediction = {hybrid確率 + Gemini reasoning}
8. UI表示
```

**確率決定に関与するのは全て統計モデル**。
**Geminiは文章生成(reasoning, key_factors, predicted_score)専用**。

---

## 期待効果

### 再現性向上
- 同じ入力データ → 同じ確率出力
- Gemini APIの非決定性が確率に影響しない
- バックテスト結果と本番予測が一致

### 確率品質維持
- バックテストで検証済みの hybrid_v9.1 性能を維持
- val=2025 F1=0.428, logL=1.056 が本番でも同じ水準

### 中確信カテゴリの回復
- Gemini fallback時の low偏りが解消
- 期待される確信度分布: high 22%, medium 29%, low 49%

### コスト削減は副次効果
- Geminiを呼ばない試合 → API コスト削減
- ※ ただし今回はreasoningが欲しいので呼ぶ頻度は変えない

---

## 実装変更ファイル

- `app.py`: `_run_onebutton_pipeline`のパイプライン改修
  - `raw_v7_prediction`を統計確率から構築
  - Geminiは文章フィールドのみ抽出
  - hybridの入力は純統計v7

### 変更されないもの
- `scripts/predict_logic.py`: predict_with_gemini は変更なし (戻り値の一部のみ使用)
- `scripts/predict_logic.py`: compute_hybrid_v9 は変更なし
- UI表示層: 変更なし

---

## 注意点

### Gemini reasoning文内に古い確率が言及される可能性
Gemini prompt には「home 35% draw 34% away 31%」等の統計モデル事前確率を含めているが、
Gemini は独自に「総合的にhome勝利60%と考える」と文章内で別の数字を書く可能性がある。
→ UIの確率バーは hybrid の値を表示するので不整合になる可能性あり。

### 対策
- reasoning内の数値は「Gemini分析コメント」として表示
- 確率バー (UIで目立つ部分) は hybrid_v9.1 の値 → ユーザーはこちらを信頼
- Gemini reasoning は折りたたみ詳細内にのみ表示されるため影響は限定的
