# ELOランタイム確認項目 (elo_runtime_checklist.md)

> 作成日: 2026-04-08

---

## チェックリスト

### 1. データフロー
- [x] cached_past_results(division) が完了試合のみ返す
- [x] cached_elo_ratings(division) がratings dictを返す (pickle互換)
- [x] get_elo_scores(division, home, away) が期待勝率(0-1)を返す
- [x] TTL=1800秒でキャッシュ管理

### 2. リーク防止
- [x] ELOは「試合終了」フィルタ済みの試合のみで構築
- [x] 予測対象試合は未完了なのでELOに含まれない
- [x] get_past_results() は date < today フィルタあり

### 3. 状態安全性
- [x] 同一セッション内でELOは不変 (cache_data)
- [x] セッション再実行時にTTL経過後再構築
- [x] EloSystem.ratings はdict.copy()で返す (元オブジェクト非共有)

### 4. fallback
- [x] ELO未取得時: elo_home_score=None → 中立値(0.5, 0.5)
- [x] 過去試合0件時: 全チーム初期値1500 → home_bonus分のみ差異

### 5. UI互換性
- [x] calculate_parameter_contributions() のeloパラメータ追加
- [x] app.pyの3箇所の呼び出しにelo引数追加
- [x] advantage_to_probs() のシグネチャ変更なし
- [x] 確率出力 (h%, d%, a%) 合計=100%
