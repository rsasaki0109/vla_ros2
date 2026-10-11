# オフライン評価と VLM 成功判定の実フレーム較正 (2026-10-11)

## 結論

- `scripts/eval_offline_lerobot.py` で記録デモに対する open-loop 評価ができるようになった。
  `smolvla_base` は episode 0 で MAE 39.8° と、現関節を保持するだけの baseline (4.6°) より大幅に悪い。
- 原因: `smolvla_base` の出力は正規化空間 (±1 程度) で、カメラキーも `camera1..3` を期待している
  (データセットは `top`/`wrist`)。SO-100 で使うには fine-tune が前提。Gazebo で base がほぼ動かないのもこれで説明できる。
- VLM 成功判定は実フレームでは使えない段階。SmolVLM2-500M / 2.2B とも done / not-done を分離できなかった。

## 確認済み事実 (CPU 4 cores)

- offline eval: `lerobot/smolvla_base`, episode 0, stride 150 (3 frames), horizon 50
  - policy MAE 39.80, first-step 40.00 / hold-state MAE 4.61, first-step 1.92
  - 予測例: state `[4.75, -86.66, 77.34, 72.86, 82.27, 0.48]` に対し pred[0] `[0.26, -0.40, -1.24, 0.05, 1.88, -0.77]`
  - 推論 10.4 s/chunk, load 150 s
- VLM 較正 (`vlm_calibrate_lerobot.py`, episodes 0–2, 最初のフレーム=not done, 最後=done)
  - 500M top default: AUROC 0.11
  - 500M wrist default: AUROC 0.22
  - 500M top goal-state prompt: AUROC 0.11
  - 500M top cube-relation prompt: AUROC 0.67
  - 2.2B top default: AUROC 0.56 (done / not-done の平均 P(yes) がともに 0.643)
  - 目視でラベルは正しい (最後のフレームは赤が青の上)。top カメラでは積んだ状態が「赤い四角＋青い縁」にしか見えない。
- 結果: `docs/assets/offline_eval_smolvla_base_ep0_cpu.json`, `docs/assets/vlm_success_real_frames_cpu.json`

## 未確認/要確認項目

- [ ] fine-tune 済み checkpoint の held-out episode での offline MAE (20k は全 episode で学習済み)
- [ ] GPU でエピソード数を増やした VLM 較正、より大きな VLM、2 フレーム (開始 vs 現在) プロンプト
- [ ] Colab での notebook 02–04 実行 (ローカル CPU では縮小設定で実行確認のみ)

## 次アクション

1. notebook 03 で held-out episode を除いて fine-tune → base / fine-tuned / hold-state を比較
2. notebook 04 を GPU で回し、AUROC が 0.5 を明確に超える VLM / プロンプトを探す

## 関連ファイル

- `src/vla_ros2/eval_offline.py`, `scripts/eval_offline_lerobot.py`
- `src/vla_ros2/vlm/success.py` (`calibrate_threshold`), `scripts/vlm_calibrate_lerobot.py`
- `notebooks/01_quickstart.ipynb` – `04_vlm_success.ipynb`
