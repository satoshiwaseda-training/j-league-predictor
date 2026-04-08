# 死にパラメータ フォローアップ (dead_parameters_followup.md)

> 作成日: 2026-04-08

---

## 現時点の方針

3ロジット本番統合と同時に死にパラメータの削除は行わない。
理由: 変更のスコープを最小に保ち、影響を分離するため。

## 状態整理

### calculate_parameter_contributions() 内の死にパラメータ
以下6パラメータは引数で受け取るが、app.pyから渡されるデータは常にNone/空dict。
結果として home_score=away_score=0.5 (中立) が出力され、contribution=0。

| パラメータ | 重み | 状態 |
|-----------|------|------|
| set_piece_conversion | 0.0204 | inactive (データ供給元なし) |
| match_day_motivation | 0.0204 | inactive |
| tactical_adaptability | 0.0204 | inactive |
| player_availability_impact | 0.0306 | inactive |
| match_trend | 0.0306 | inactive |
| referee_tendency | 0.0306 | inactive |

### 3ロジット統合への影響
- 死にパラメータは `raw_home_advantage` と `closeness` に寄与しない
  (home_advantage=0 なので raw_adv への加算は0)
- 3ロジット変換自体には影響なし
- ただし raw_adv の絶対値が本来より小さくなる (有効重みの分母が大きいため)
  → closeness が実態より高めに出る可能性

### 次ループでの対応候補
1. MODEL_WEIGHTS の死にパラメータ6個を 0.0000 に設定
2. calculate_parameter_contributions() で None/空の場合はスキップするロジック追加
3. closeness の算出で有効パラメータのみを使うよう修正
