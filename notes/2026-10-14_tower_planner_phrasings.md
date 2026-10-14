# タワープランナー: 例に無い言い回し (2026-10-14)

## 結論

`hf_tower` の出力を「下から上の色リスト」から「ゴールに出てくる順の色リスト + その向き
(`top_down`/`bottom_up`)」に変え、反転はコード (`parse_tower`) で行うようにした。モデルは
並べ替えをせず書き写すだけになり、全モデルで改善 (1.5B 122→131/140, 0.5B 67→119, 3B 113→128)。
few-shot 例を 2 つ増やすのはどちらの形式でも悪化した。

## 確認済み事実 (CPU 4 cores, 20 タワー × 7 言い回し = 140 問)

- 言い回し: 例と同型 3 種 (bottom_up, top_down, chain) + 例に無い 4 種 (top_list, bottom_list,
  on_top_of, top_sentences)。評価の色は例の色と重複なし。
- 1.5B, 正解数 / 140 (例に無い 4 種の内訳):
  - hf (手順を直接): 43
  - 下から上リスト (#18 の hf_tower): 122 (top_list 11/20)
  - 同 + 例 2 つ追加: 112
  - 書かれた順 + 向き (新 hf_tower): 131 (top_list 20, bottom_list 20, on_top_of 16, top_sentences 18)
  - 同 + 例 2 つ追加: 125 (top_down 10/20 に悪化)
- 0.5B: hf 21, 下から上 67 (top_down 0/20, top_sentences 0/20), 新 hf_tower 119 (top_sentences 3/20)。
- 3B: hf 54, 下から上 113 (top_list 5/20), 新 hf_tower 128 (top_list 14/20)。
- 新 hf_tower の失敗は主に向きラベルの誤り (例: "Put gray on top of blue, blue on top of red."
  を top_down としつつ色を下から並べる)。
- 3B の初回試行は 140 問すべてパース失敗と出たが、計測中に `parse_tower` を新形式へ書き換えた
  ため試作ハーネス側が壊れていただけ。保存した応答を新しい parser で採点し直すと 128、
  リポジトリのコードで再計測しても 128 で一致。
- s/問 は同じモデルでも実行ごとに変わる (3B: 6〜11 s)。新形式は出力が長いぶんやや遅い。

## 未確認/要確認項目

- [ ] 単一タワー以外のゴール (複数タワー、並べる、片付け)
- [ ] GPU で 7B 以上
- [ ] 向きの自己矛盾 (ラベルと色順の不整合) を検出して再質問する

## 関連ファイル

- `src/vla_ros2/planner.py` (`TOWER_SYSTEM_PROMPT`, `TOWER_EXAMPLES`, `parse_tower`)
- `src/vla_ros2/planner_eval.py` (`SEEN_PHRASINGS`, `HELD_OUT_PHRASINGS`)
- `docs/assets/planner_tower_eval_cpu.json`
