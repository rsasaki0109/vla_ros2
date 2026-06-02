## VLA Method Profiles

| Method | Family | Role | Inputs | Action | Chunks | Local | Remote | Status |
|---|---|---|---|---|---|---|---|---|
| `dummy` | dry-run baseline | CI/no-GPU runtime sanity check | image optional<br>instruction optional<br>state optional | eef_delta (7,): zero 7-DoF end-effector delta | optional via chunk_size | supported | supported | available |
| `groot` | humanoid/generalist foundation model | experimental humanoid/generalist adapter target | multimodal observations<br>instruction/task context<br>robot state expected | custom adapter-specific: humanoid/generalist action interface | adapter-specific | experimental placeholder | recommended | experimental |
| `openvla` | VLA foundation model | single-image VLA reference adapter | single RGB image<br>natural language instruction<br>optional unnormalization key | eef_delta (7,): OpenVLA-style 7-DoF action | no | supported with optional ML dependencies | recommended for robot-side ROS2 | missing optional deps: pip install "vla_zoo[openvla]" |
| `pi0` | pi-family VLA | remote-first action-chunk VLA target | images per policy config<br>natural language instruction<br>robot state expected | eef_delta (7,) placeholder: policy-specific manipulation action | expected | not implemented in MVP | recommended | experimental |
| `random` | stochastic baseline | action plumbing and visualization stress check | instruction optional<br>image optional<br>state optional | eef_delta (7,): seeded random 7-DoF end-effector delta | no | supported | supported | available |
| `scripted` | rule-based baseline | upper-bound sanity check for the scripted smoke scene | phase metadata<br>instruction optional<br>image optional | eef_delta (7,): phase-aware 7-DoF end-effector delta | no | supported | supported | available |
| `smolvla` | LeRobot policy | multi-camera/state/action-chunk adapter target | multi-camera images<br>natural language instruction<br>robot state | eef_delta (7,) placeholder: policy-specific action chunk | expected | not implemented in MVP | recommended | experimental |

These profiles describe runtime integration shape, not model quality. External model weights, datasets, and licenses are not redistributed by vla_zoo.
