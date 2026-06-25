# NPU Op Optimization Log

This log records each successful NPU optimization rule or meaningful debugging
milestone. Add one entry whenever an op/fallback pattern gains support or a
model produces a verified improvement.

## Entry Template

### YYYY-MM-DD - Short title

- Op/Fallback Type:
- Models:
- Failure Symptom:
- Root Cause Hypothesis:
- Implementation Method:
- Debug Process:
- Vela Before/After:
- Board Verification:
- Remaining Risk:

## Initial Rules

### 2026-06-25 - Baseline safe TFLite rewrites

- Op/Fallback Type: FLOAT16 DEQUANTIZE constants, float PReLU islands, invalid INT8 Conv/DepthwiseConv bias tensors, missing serving_default signatures
- Models: Initial automation targets low-NPU models such as retinaface_mnet_face_160 and segformerb0_seg_512
- Failure Symptom: Vela leaves unsupported or poorly quantized regions on CPU, reducing NPU utilization
- Root Cause Hypothesis: Some fallback islands come from graph representation details that can be normalized without changing model intent
- Implementation Method: Register safe TFLite rewrite rules and let the NPU optimizer select candidates based on Vela before/after metrics
- Debug Process: Run baseline Vela, apply candidate rewrites, re-run Vela, compare CPU/NPU operator counts and fallback types
- Vela Before/After: Filled by optimizer_trace.json for each model run
- Board Verification: Required; optimization is not considered fully successful until benchmark firmware reports status=ok and improved NPU/fallback metrics
- Remaining Risk: PC-side Vela improvement can still fail board Flash/SRAM/runtime constraints, so baseline artifacts are preserved for fallback
