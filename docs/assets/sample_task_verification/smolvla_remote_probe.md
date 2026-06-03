# Remote VLA Probe: smolvla

Health-first remote runtime probe. This records a single `/v1/predict`
response over HTTP. It is not a robot task-success benchmark.

- model: `smolvla`
- remote_url: `http://127.0.0.1:8011`
- instruction: `pick up the red block`
- status: `ok`

## Health

```json
{
  "ready": true,
  "model": "smolvla",
  "runtime": "server",
  "status": "ok"
}
```

## Recorded Action

```json
{
  "action_space": "custom",
  "data": [
    -0.11001584678888321,
    0.1927943378686905,
    0.09785352647304535,
    0.10416978597640991,
    0.23979687690734863,
    -0.1279381513595581
  ],
  "shape": [
    6
  ],
  "names": [],
  "frame_id": null,
  "control_hz": null,
  "normalized": false,
  "dt": null,
  "confidence": null,
  "chunk_index": null,
  "metadata": {
    "model": "lerobot/smolvla_base",
    "adapter": "SmolVLAAdapter",
    "image_keys": [
      "observation.images.camera1",
      "observation.images.camera2",
      "observation.images.camera3"
    ],
    "state_key": "observation.state",
    "filled_images": [
      "observation.images.camera1<=primary",
      "observation.images.camera2<=primary",
      "observation.images.camera3<=primary"
    ],
    "state_filled_or_resized": true,
    "latency_ms": 2759.0965640265495
  }
}
```
