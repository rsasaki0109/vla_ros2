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

- [ ] Gazebo CI（nightly / self-hosted）への組み込み
- [ ] GUI モード（`-s` なし）での目視確認
- [ ] 他 ROS stack 稼働中の default domain での安定性

## 次アクション

1. `git push`（未 push commits あり）
2. controller bridge 雛形（実機 Phase C 用）
3. SmolVLA fine-tune または実機 bring-up Phase C

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
