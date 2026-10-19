# On-device model artifacts (P5)

The from-scratch device SLM runs on-device via **Core ML** (Neural Engine / GPU). Because the
model is tiny and char-level, we export the **forward pass** (tokens → logits) and run the greedy
decode loop in Swift; the char tokenizer + delimiters travel alongside as JSON.

## Export
```bash
pip install coremltools torch
python tools/export_coreml.py --ckpt dcslm/out/ckpt.pt --out apple/DeviceControlKit/Models
# -> DCSLM.mlpackage  +  dcslm_tokenizer.json
```
`scratch/model.py` has an `EXPORT_MODE` (static shapes, plain attention, no dynamic RoPE slice)
that the exporter enables for a clean conversion graph. The **tracing step is validated on
Linux**; producing the final `.mlpackage` is finalized on macOS with a matching coremltools —
Core ML's Torch converter is version-sensitive (a `aten::Int` cast quirk was hit on
coremltools 9.0 on Linux). Options if it recurs: pin an older coremltools, or route via ONNX
(`torch.onnx.export` → `coremltools`).

## Runtime wiring (Swift)
On device, the model conforms to `LanguageModel` (see `DeviceAgent.swift`): build a
`[1, block_size]` int window (right-padded), run the Core ML model, read logits at the last real
position, greedy-decode char-by-char from `<request>` until ``, then hand the
DSL string to `DeviceControlEngine`.

## MLX alternative
The **fine-tuned open-base** models (Qwen/SmolLM2) quantize to 4-bit MLX directly via `mlx_lm`
(see `finetune/README.md`) — that path is the shipping option if we choose a pretrained base over
the from-scratch model (which the OOD results favor for robustness).

## Not committed
`.mlpackage` / `.mlmodelc` binaries are git-ignored — regenerate with the export script.
