# Deployment Benchmarks

**Test hardware:** arm, Darwin 25.3.0 (CPU-only inference for all configs)

| Config | Latency (ms) | FPS | Model size (MB) |
|---|---|---|---|
| PyTorch FP32 | 18.75 | 53.33 | 5.94 |
| ONNX FP32 | 19.18 | 52.14 | 11.64 |
| ONNX INT8 | 30.84 | 32.42 | 3.14 |
