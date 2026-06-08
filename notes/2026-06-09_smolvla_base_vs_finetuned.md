# SmolVLA base vs fine-tuned 比較 (2026-06-09)

## 結論

同一 episode（`svla_so100_stacking` ep0）で offline + Gazebo 比較を実施。

- **offline**: ポリシー差が GIF で明確（fine-tuned は大振幅 action）
- **Gazebo**: runtime 起動・`/vla/action` publish は **修正後 PASS**（42–44 msgs/run）
- **Gazebo 関節**: 依然として実位移 ~0 rad — bridge / action scale 要調査

## 確認済み事実

- 根因1: `vla_runtime_node` が system Python で SmolVLA import 失敗 → `scripts/vla_gz_env.sh` で venv shebang パッチ
- 根因2: 録画が model load 前に終了 → `--warmup-sec 180` で ready + action 待ち
- 根因3: `pkill -f vla_smolvla` が `record_gz_smolvla_*` 自身を kill → パターン修正
- `./scripts/gz_smolvla_validate.sh infer` — **PASS**（2026-06-09 再実行）
- Gazebo metrics（60s capture）:
  - base: `actions_received=42`, `max_joint_delta_rad≈8e-11`
  - fine-tuned: `actions_received=44`, `max_joint_delta_rad≈2e-9`

## 未確認/要確認項目

- [ ] joint bridge の action scale / trajectory が Gazebo 関節を動かす条件
- [ ] Playground × Gazebo ブラウザ E2E

## 次アクション

1. `vla_smolvla_joint_bridge` の action→trajectory スケール調整
2. Playground から fine-tuned stacking 目視
3. Gazebo compare GIF 再録画（actuation 修正後）

## 関連ファイル

- `vla_zoo/scripts/vla_gz_env.sh`
- `vla_zoo/scripts/record_gz_smolvla_compare.sh`
- `vla_zoo/docs/assets/gz_smolvla_compare_metrics.json`
