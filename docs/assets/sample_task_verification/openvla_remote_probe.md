# Remote VLA Probe: openvla

Health-first remote runtime probe. This records a single `/v1/predict`
response over HTTP. It is not a robot task-success benchmark.

- model: `openvla`
- remote_url: `http://127.0.0.1:8012`
- instruction: `pick up the red block`
- status: `ok`

## Health

```json
{
  "ready": true,
  "model": "openvla",
  "runtime": "server",
  "status": "ok"
}
```

## Recorded Action

```json
{
  "action_space": "eef_delta",
  "data": [
    0.004488371778279543,
    0.0018421962158754468,
    1.8581469703349285e-05,
    -0.015465142205357552,
    -0.014177892357110977,
    -0.05164425075054169,
    0.9960784316062927
  ],
  "shape": [
    7
  ],
  "names": [],
  "frame_id": null,
  "control_hz": null,
  "normalized": false,
  "dt": null,
  "confidence": null,
  "chunk_index": null,
  "metadata": {
    "model": "openvla/openvla-7b",
    "adapter": "OpenVLAAdapter",
    "unnorm_key": "bridge_orig",
    "latency_ms": 2916.4278220850974
  }
}
```
