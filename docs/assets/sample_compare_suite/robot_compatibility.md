## VLA Robot Compatibility

Robot profile: `single-camera-eef`

- cameras: 1
- state: False
- action chunks: False
- action spaces: eef_delta, eef_pose, gripper
- domains: manipulation

| Model | Fit | Adapter status | Score | Action | Issues | Next step |
|---|---|---|---:|---|---|---|
| `openvla` | compatible | available | 100 | eef_delta (7,) | - | Run the adapter behind a remote GPU server for robot-side ROS2 deployment. |
| `pi0` | blocked | experimental | 20 | custom checkpoint-specific; lerobot/pi0 is (6,), lerobot/pi0_base is (32,) | warning: adapter expects robot state; output quality or schema may be invalid without it<br>error: adapter outputs 'custom'; robot profile supports eef_delta, eef_pose, gripper<br>error: adapter expects action chunks but robot profile consumes single actions | Provide proprioception in VLAObservation.state or ROS joint state inputs. |
| `smolvla` | blocked | missing optional deps: pip install "vla_zoo[smolvla]" | 0 | custom checkpoint-specific; lerobot/smolvla_base is (6,) | error: adapter expects at least 2 camera stream(s); robot profile declares 1<br>error: adapter requires robot state/proprioception<br>error: adapter outputs 'custom'; robot profile supports eef_delta, eef_pose, gripper | Add the required camera streams or remap the adapter image inputs. |
| `groot` | blocked | experimental | 0 | custom adapter-specific | error: adapter expects at least 2 camera stream(s); robot profile declares 1<br>warning: adapter expects robot state; output quality or schema may be invalid without it<br>error: adapter outputs 'custom'; robot profile supports eef_delta, eef_pose, gripper<br>error: adapter domain 'humanoid/generalist' does not match robot domains manipulation | Add the required camera streams or remap the adapter image inputs. |

This is a deployment-shape check. It does not validate model quality, calibration, safety, or real robot task success.
