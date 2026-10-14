# ゴールプランナー (2026-10-12)

## 結論

1 つのゴールからシーケンサの手順リストを作る `vla_ros2.planner` を追加。小型ローカル LLM
(Qwen2.5 0.5B/1.5B/3B) は「下から順に」書かれたゴールしか正しく分解できず、上から書かれた
ゴールでは順序を逆にする。既定は `planner:=static` (ゴールをそのまま 1 手順) のまま。

## 確認済み事実 (CPU 4 cores, 20 タワー × 3 言い回し = 60 問, 正解は 1 通り)

- 0.5B 16/60 (bottom-up 11/20, top-down 1/20, chain 4/20), 1.8 s/問
- 1.5B 29/60 (19/20, 3/20, 7/20), 4.6 s/問
- 3B 33/60 (20/20, 2/20, 11/20), 8.6 s/問
- JSON パース失敗は全モデル 0。
- 失敗例: "Make a stack with blue on top, red under it." → "Put the red cube on the blue cube."
- system prompt に「下から積む」規則を明記すると悪化 (1.5B 25/60, 3B 27/60) → revert。
- 評価の色 (red/green/blue/gray) はプロンプト例の色 (yellow/purple/orange/pink/white) と重複なし。

## 未確認/要確認項目

- [ ] GPU で 7B 以上のモデル
- [x] LLM には下から上の順番 (JSON) だけを出させ、手順はコードで生成する方式 → `notes/2026-10-14_tower_planner.md`

## 関連ファイル

- `src/vla_ros2/planner.py`, `src/vla_ros2/planner_eval.py`, `scripts/planner_eval.py`
- `docs/assets/planner_tower_eval_cpu.json`
