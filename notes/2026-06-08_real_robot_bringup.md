# Real-robot bring-up validation (2026-06-08)

## 結論

Phase A（smoke launch test）と Phase B（live I/O dry-run）は **PASS**。
このマシンの ROS グラフは dashcam 系（`/camera/image_raw`）で、**アーム用 `/joint_states` パブリッシャは未確認**。
Phase C 以降（controller bridge / actuation）は **未実施**。

## 確認済み事実

- `./scripts/bringup_validate.sh all` — Phase A: colcon smoke 2 tests OK
- Phase B: `vla_runtime_node` + `bringup.dashcam.example.yaml` で `ready: true`, `status_text: dry_run: action suppressed`
- カメラ: `/camera/image_raw` @ ~10 Hz（publisher: `dashcam2traj_node`, QoS RELIABLE）
- `/joint_states`: subscriber のみ（publisher count 0）
- `ros2 topic pub` のデフォルト QoS は `instruction_qos()`（TRANSIENT_LOCAL）と非互換 → `scripts/publish_instruction.py` で解決

## 未確認/要確認項目

- [ ] `/joint_states` を出す arm stack の接続
- [ ] Phase C: `publish_actions_in_dry_run:=true` + controller bridge のパース検証
- [ ] Phase D: `dry_run:=false` + 実機 actuation（E-stop / clip bounds 確認後）
- [ ] `ros2 topic pub` の `--qos-durability transient_local` が CLI 版でも常に効くか（WARN は出るが inference は進むケースあり）

## 次アクション

1. アーム stack を起動し `/joint_states` publisher を確認
2. `robot.example.yaml` をコピーして topic remap + clip を調整
3. Phase C で bridge のみテスト → Phase D は E-stop 付きで低 `control_hz`

## 実行コマンド

```bash
cd /media/sasaki/aiueo/ai_coding_ws/vla_zoo
./scripts/bringup_validate.sh all

# 手動 Phase B
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch vla_ros2 dummy.launch.py \
  params_file:="$PWD/ros2/vla_ros2/config/bringup.dashcam.example.yaml"
python3 scripts/publish_instruction.py --text "pick up the cup" --repeat-hz 2
```

## 関連ファイル

- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/ros2/BRINGUP.md`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/ros2/vla_ros2/config/bringup.dashcam.example.yaml`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/scripts/bringup_validate.sh`
- `/media/sasaki/aiueo/ai_coding_ws/vla_zoo/scripts/publish_instruction.py`
