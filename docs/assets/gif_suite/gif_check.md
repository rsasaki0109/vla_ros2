## PyBullet GIF Check

- manifest: `docs/assets/gif_suite/gif_manifest.json`
- status: ok
- assets: 9

This QA report checks that generated GIFs exist, decode, have enough frames, use the expected resolution, are not low-variance blank files, and match the manifest. It does not validate VLA model quality or real robot behavior.

| GIF | Status | Frames | Resolution | Size | Issues |
|---|---|---:|---|---:|---|
| `simulation_pick_red_block_dummy.gif` | ok | 60 | 960x540 | 1645249 | - |
| `simulation_pick_red_block_scripted.gif` | ok | 60 | 960x540 | 1653805 | - |
| `simulation_pick_red_block_random.gif` | ok | 60 | 960x540 | 1645485 | - |
| `simulation_move_red_block_left_dummy.gif` | ok | 60 | 960x540 | 1632011 | - |
| `simulation_move_red_block_left_scripted.gif` | ok | 60 | 960x540 | 1638930 | - |
| `simulation_move_red_block_left_random.gif` | ok | 60 | 960x540 | 1632805 | - |
| `simulation_move_red_block_right_dummy.gif` | ok | 60 | 960x540 | 1655339 | - |
| `simulation_move_red_block_right_scripted.gif` | ok | 60 | 960x540 | 1663123 | - |
| `simulation_move_red_block_right_random.gif` | ok | 60 | 960x540 | 1656960 | - |
