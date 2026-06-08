# Gazebo closed-loop validation (2026-06-08)

## 結論

Gazebo Sim 閉ループ検証 **Phase 1 + Phase 2 PASS**。
`/vla/action` → `vla_action_bridge_node` → `joint_trajectory_controller` まで通る。

## 確認済み事実

- `./scripts/gz_smoke_validate.sh 1` — runtime graph: `/vla/action`, `ready` status, `dummy` adapter
- `./scripts/gz_smoke_validate.sh 2` — `random` adapter + actuation: controllers active, `trajectories>=1`
- コントローラ起動: `gz_arm.launch.py` で 5s 遅延 + 単一 spawner `--activate-as-group`
- 隔離: ランダム `ROS_DOMAIN_ID`（他 stack との衝突回避）
- 既知: `joint_1` 実位移は微小（~1e-5 rad）— bridge trajectory で actuation path を確認

## 未確認/要確認項目

- [x] Gazebo nightly CI（`.github/workflows/gazebo-nightly.yml`）
- [ ] GUI モード（`-s` なし）での目視確認
- [ ] 他 ROS stack 稼働中の default domain での安定性

## 次アクション

1. ~~SmolVLA fine-tune（20k steps）→ GIF 差し替え~~ 5k checkpoint で GIF 更新済み（20k 完走後に `./scripts/record_finetuned_gz_demo.sh` で再録画可）
2. 実機 bring-up Phase C（`/joint_states` 接続後）

## 実行コマンド

```bash
cd /media/sasaki/aiueo/ai_coding_ws/vla_zoo
./scripts/gz_smoke_validate.sh all
```

## 関連ファイル

- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/ros2/SIM.md`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/scripts/gz_smoke_validate.sh`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/scripts/gz_smoke_probe.py`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/ros2/vla_ros2_gz/launch/gz_arm.launch.py`
