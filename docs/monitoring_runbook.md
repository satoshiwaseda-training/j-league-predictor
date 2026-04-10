# Primary モデル監視ランブック (monitoring_runbook.md)

> 作成日: 2026-04-11

---

## 目的

hybrid_v9.1 を primary に昇格後、運用中の劣化を早期検知し、
必要に応じて v7 baseline に自動降格するための監視体制。

---

## 監視の仕組み

### ローリングウィンドウ
| ウィンドウ | サイズ | 用途 |
|-----------|--------|------|
| short | 20試合 | 早期警戒 |
| medium | 40試合 | 中期トレンド + 自動降格判定 |
| formal | 80試合 | 正式定着判定 |

### 各ウィンドウで比較する指標
- accuracy
- F1 macro
- log loss
- draw予測数

### 比較対象
- **Primary**: hybrid_v9.1 (本番UI表示)
- **Baseline**: v7 refined (fallback)
- **Shadow**: v8.1 (内部ログ)

各予測は3種同時に `data/predictions.json` に保存される。

---

## 警戒レベル

| レベル | 条件 | 意味 |
|--------|------|------|
| GREEN | primary優位または同等 | 正常 |
| YELLOW | 1指標でbaseline優位(>=5pp gap) | 注意 |
| ORANGE | 2指標でbaseline優位 | 警告 |
| RED | 3指標すべてでbaseline優位 | 危険 |

### 警戒判定の閾値
```python
DOWNGRADE_THRESHOLDS = {
    "accuracy_gap":  -0.05,  # primary - baseline <= -5pp → 警戒
    "f1_gap":        -0.05,  # 同上
    "log_loss_gap":   0.05,  # primary - baseline >= +0.05 → 警戒
}
```

### 最小サンプル数
- 警戒判定: n >= 20
- 自動降格: n >= 40

---

## 自動降格条件

### 降格判断
以下のいずれかを満たしたら v7 にfallback推奨:

1. **medium/formal ウィンドウで RED 警戒** (n>=40)
2. **RED 警戒が連続3回以上** (継続監視で観測)

### 降格手順
```bash
# 1. 監視レポート実行
python scripts/monitoring.py

# 2. auto_downgrade.should == True を確認

# 3. scripts/predict_logic.py を編集
# PRIMARY_MODEL_VERSION = "hybrid_v9.1"  # 降格前
# ↓
# PRIMARY_MODEL_VERSION = "v7_refined"   # 降格後

# 4. コミット・push
git add scripts/predict_logic.py
git commit -m "rollback: hybrid_v9.1 → v7 (monitoring降格)"
git push
```

### 昇格復帰
問題が解決し、再度 hybrid_v9.1 が primary 候補となる場合:
1. 改善内容をコミット
2. `PRIMARY_MODEL_VERSION = "hybrid_v9.1"` に戻す
3. 再度監視を開始

---

## スキーマ仕様

### `data/predictions.json` 各エントリ (v2 schema)
```json
{
  "id": "8-char hex",
  "_key": "date_home_away",
  "saved_at": "ISO datetime",
  "schema_version": "v2",
  "division": "j1",
  "match": {"date", "home", "away", "venue", ...},

  "model_version": "hybrid_v9.1",
  "baseline_model_version": "v7_refined",
  "role": "primary",

  "prediction": {
    "home_win_prob": int,
    "draw_prob": int,
    "away_win_prob": int,
    "predicted_score": "X-Y",
    "confidence": "high" | "medium" | "low",
    "pred_winner": "home" | "draw" | "away",
    "model": "gemini-2.5-flash" | "statistical-only",
    "hybrid_selection": "v7" | "skellam" | "weighted"
  },

  "baseline_prediction": {
    "home_win_prob": int,
    "draw_prob": int,
    "away_win_prob": int,
    "pred_winner": str,
    "model_version": "v7_refined"
  },

  "shadow_prediction": {
    "home_win_prob": int,
    "draw_prob": int,
    "away_win_prob": int,
    "pred_winner": str,
    "model_version": "v8.1_shadow"
  },

  "actual": {"score": "X-Y", "winner": str} | null
}
```

---

## 運用ルーティン

### 毎週末 (試合後)
1. 成績タブで結果を登録 (既存UI)
2. `python scripts/monitoring.py` を実行
3. warnings リストを確認
4. auto_downgrade.should が True なら降格手順へ

### 定期 (月1回)
1. monitoring_latest.json を確認
2. 直近short/medium/formal ウィンドウの推移を比較
3. n>=80 到達時に正式定着を判断

---

## トラブルシューティング

### 予測が legacy スキーマのまま
- 原因: 昇格前の予測データ
- 対応: 新規予測から v2 スキーマが自動適用される (データ変換は不要)

### baseline 不在で警戒判定できない
- 原因: 昇格直後で baseline_prediction がまだ保存されていない
- 対応: 次の予測実行後から比較可能 (ワンボタン予測タブで実行)

### RED警戒が出た
- 原因: hybrid_v9.1 が baseline より明確に劣化
- 対応:
  1. まず直近20試合の試合を確認 (特定のリーグ状況か?)
  2. 2026シーズン固有のパターンか継続的な劣化かを判断
  3. 継続的な劣化なら降格手順を実行
