# Adapter Contract

Adapters isolate VLA model code from the runtime. They should subclass `VLAAdapter`, implement `from_config()`, and implement `predict_observation()`.

```python
class MyAdapter(VLAAdapter):
    name = "my_adapter"
    model_id = "vendor/model"
    action_spec = ActionSpec(action_space="eef_delta", shape=(7,))

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)

    def predict_observation(self, observation):
        ...
```

## Requirements

- Return `VLAAction` or `VLAActionChunk`, never a raw array.
- Declare `ActionSpec` accurately.
- Use lazy imports for heavy dependencies.
- Raise `MissingDependencyError` with an install hint for optional extras.
- Do not download weights in tests.
- Keep robot-specific actuation outside the adapter.

## Entry Points

```toml
[project.entry-points."vla_zoo.adapters"]
my_robot_vla = "my_pkg.adapters:MyRobotVLAAdapter"
```

## Metadata

Each adapter should document input cameras, proprioception needs, action names, control rate, chunk size, remote support, dependency status, and model license caveats.

Built-in profiles use `AdapterInfo.metadata` keys that external adapters can also provide:

| Key | Meaning |
|---|---|
| `family` | Method family, such as dry-run baseline, VLA foundation model, or LeRobot policy |
| `compare_role` | Why this method appears in comparison tables |
| `input_requirements` | Tuple/list of required observations, cameras, instruction, and state |
| `output` | Human-readable action output description |
| `action_space` | Runtime action space, matching `ActionSpec.action_space` when possible |
| `action_shape` | Expected action tensor shape |
| `control_hz` | Expected outer-loop policy rate |
| `action_chunks` | Whether the method emits action chunks |
| `proprioception` | Whether robot state is required |
| `local_runtime` | Local adapter support status |
| `remote_runtime` | Remote inference support status |
| `dependency_profile` | Optional dependency and serving environment summary |
| `license_caveat` | External project, model, dataset, or checkpoint license notes |

These fields are displayed by:

```bash
vla-zoo compare methods
```

Generate the per-adapter capability cards from the same registry metadata:

```bash
vla-zoo compare cards --out-dir docs/adapters
```

See [docs/adapters/README.md](adapters/README.md) for the current cards.
