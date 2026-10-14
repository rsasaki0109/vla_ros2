# タワー順序プランナー (2026-10-14)

## 結論

LLM に手順文ではなく「下から上の色の並び」(JSON 配列) だけを出させ、手順はコードで生成する
`HFTowerPlanner` (`planner:=hf_tower`) で、タワー目標の正解率が全モデルで 2 倍前後に上がった。
効いているのは出力形式で、few-shot 例を揃えた手順出力 (`--examples tower`) は改善しない。
1.5B が最良 (63/80) かつ 3B より速いので、シーケンサの `planner_model` 既定を 1.5B にした。
`planner` の既定は `static` のまま。

## 確認済み事実 (CPU 4 cores, 20 タワー × 4 言い回し = 80 問, 正解は 1 通り)

| モデル | hf (既存例) | hf (`--examples tower`) | hf_tower | ペア方式 (試作) |
|---|---:|---:|---:|---:|
| 0.5B | 17 | 16 | 39 | 14 |
| 1.5B | 29 | 33 | 63 | 18 |
| 3B | 35 | 32 | 54 | 34 |

- hf_tower 1.5B: bottom-up 20/20, top-down 16/20, chain 16/20, top_list 11/20, 3.3 s/問。
- hf_tower 3B: 20/20, 15/20, 14/20, 5/20, 5.9 s/問。top_list では並びを反転せず書き写す。
- `top_list` ("From top to bottom, the tower is red, green, blue.") はどの few-shot 例にも無い
  言い回し。bottom_up/top_down/chain は hf_tower と `--examples tower` の例に 1 つずつ含まれる
  (色は評価と重複なし)。
- 旧 3 言い回しの hf 3B は 33/60 で前回 (2026-10-12) と一致。
- ペア方式: [upper, lower] の順不同ペアを出させてコードで整列。手順出力と同程度以下で、
  0.5B はパース失敗 36/80。リポジトリには入れていない。

## 未確認/要確認項目

- [ ] GPU で 7B 以上
- [ ] 単一タワー以外のゴール (複数タワー、並べる、片付け)
- [x] top_list 等、例に無い言い回しへの頑健性 → `notes/2026-10-14_tower_planner_phrasings.md`

## 関連ファイル

- `src/vla_ros2/planner.py` (`HFTowerPlanner`, `parse_tower`, `tower_steps`)
- `src/vla_ros2/planner_eval.py`, `scripts/planner_eval.py` (`--planner hf_tower`, `--examples tower`)
- `docs/assets/planner_tower_eval_cpu.json`
