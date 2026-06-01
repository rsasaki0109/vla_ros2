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
