## PyBullet GIF Gallery

These GIFs are rendered from the bundled PyBullet simulation. They show runtime plumbing and adapter action traces, not real robot task success.

| dummy / pick_red_block | scripted / pick_red_block | random / pick_red_block |
|---|---|---|
| ![dummy pick_red_block PyBullet simulation](simulation_pick_red_block_dummy.gif)<br>frames=60, size=1645249 bytes | ![scripted pick_red_block PyBullet simulation](simulation_pick_red_block_scripted.gif)<br>frames=60, size=1653805 bytes | ![random pick_red_block PyBullet simulation](simulation_pick_red_block_random.gif)<br>frames=60, size=1645485 bytes |

| dummy / move_red_block_left | scripted / move_red_block_left | random / move_red_block_left |
|---|---|---|
| ![dummy move_red_block_left PyBullet simulation](simulation_move_red_block_left_dummy.gif)<br>frames=60, size=1632011 bytes | ![scripted move_red_block_left PyBullet simulation](simulation_move_red_block_left_scripted.gif)<br>frames=60, size=1638930 bytes | ![random move_red_block_left PyBullet simulation](simulation_move_red_block_left_random.gif)<br>frames=60, size=1632805 bytes |

| dummy / move_red_block_right | scripted / move_red_block_right | random / move_red_block_right |
|---|---|---|
| ![dummy move_red_block_right PyBullet simulation](simulation_move_red_block_right_dummy.gif)<br>frames=60, size=1655339 bytes | ![scripted move_red_block_right PyBullet simulation](simulation_move_red_block_right_scripted.gif)<br>frames=60, size=1663123 bytes | ![random move_red_block_right PyBullet simulation](simulation_move_red_block_right_random.gif)<br>frames=60, size=1656960 bytes |

Reproduce a single GIF:

```bash
vla-zoo demo pybullet --model dummy --task pick_red_block --out docs/assets/gif_suite/simulation_pick_red_block_dummy.gif --model-call-every 8 --render-stride 8
```
