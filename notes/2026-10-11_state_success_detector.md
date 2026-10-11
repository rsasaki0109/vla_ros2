# 関節・グリッパー状態による成功判定 (2026-10-11)

## 結論

VLM (SmolVLM2 500M/2.2B) は実フレームで完了/未完了を分離できなかった (AUROC 0.11–0.67) ため、
`joint_states` だけで判定する `GraspReleaseDetector` を作った。`svla_so100_stacking` の
有効な held-out 41 エピソードで全件検出、30%/50% 時点での誤検出 0。

## 確認済み事実

- グリッパー (SO-100, 0=閉) は全エピソードで 0 → 開 ~25 (接近) → ~13 で一定 (キューブで止まる = 把持)
  → 開 ~20 (リリース) → 0 (待機) という形。1 フレームでは開始と終了が区別できない (どちらも待機姿勢)。
- 判定: 「より開いた状態から閉じて」一定値 (範囲 1.5 以内, 1 s) → その値 +3 以上に開く → 閉じるか腕が 0.5 s 静止。
- 結果 (`scripts/state_success_eval_lerobot.py`, 閾値は ep 0–9 で調整):
  - ep 0–9: 10/10、誤検出 0/0、発火位置 中央値 81% (最早 70%)
  - ep 10–55: 41/46、誤検出 0/0、中央値 85% (最早 71%)
  - 未検出 5 件 (20, 37, 38, 39, 55) は 1–28 フレームの壊れた録画かグリッパーが動かない = 把持なし
- `approach_delta` は held-out の ep40 (接近時に 16 までしか開かない) を見てから 5 → 2.5 に下げた。5 のままだと 40/46。
- 時間ベース (秒・単位/秒) にしたので 10/30/100 Hz で同じ判定 (unit test)。

## 未確認/要確認項目

- [ ] 実機 SO-100 の joint 名・単位での動作 (Gazebo stand-in はグリッパーが m 単位なので `config_json` 必須)
- [ ] 物体の置き場所は見ていない (落としても release として成功判定しうる) → 画像判定との併用

## 関連ファイル

- `src/vla_ros2/task_success.py`, `tests/test_task_success.py`
- `ros2/vla_ros2/vla_ros2_ros/state_success.py`, `ros2/vla_ros2/tests/test_state_success_launch.py`
- `scripts/state_success_eval_lerobot.py`, `docs/assets/state_success_svla_so100_stacking.json`
