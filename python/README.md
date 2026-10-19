# AgentCore (Python) — dev & eval harness

The reference implementation of the on-device agent: the tool-calling loop, tool
protocol, JSON-schema validation, safety gate, and an evaluation suite. It runs the
**exact same agent logic** we port to Swift for the app (ADR-0006), so hardening it here
directly improves the shipped product.

It has **zero third-party dependencies** for the core + eval — everything below runs on
stock Python 3.9+ on Linux, CI, or a Mac, using a **mock model backend**. Real model
inference (MLX) is an optional backend you enable on a Mac.

## Why a mock backend?
MLX cannot run in the iOS/iPadOS Simulator or on Linux/CI (it needs a modern Metal GPU —
see `../docs/design.md` §3.4). The loop is designed around a `ModelBackend` interface with
two implementations:

| Backend | Where it runs | Purpose |
| --- | --- | --- |
| `MockBackend` | anywhere (Linux/CI/Simulator) | scripted or rule-based turns; tests + eval |
| `MLXBackend` | Apple Silicon (Mac / device) | real inference via `mlx-lm` |

## Run it here (no GPU, no model)
```bash
cd python

# unit tests: loop, parser, registry, safety gate
python -m unittest discover -s tests -q

# evaluation suite (uses a rule-based reference policy)
python -m eval.runner

# interactive demo
python demo.py
python demo.py --yes "remind me to call mom at 6pm"
```

## Run it with the real model (on a Mac)
```bash
pip install "mlx-lm>=0.19"     # Apple Silicon only

# runs the SAME tasks against the real SLM and reports the model's real numbers
python -m eval.runner --backend mlx
python demo.py --backend mlx --yes "set a timer for 5 minutes"
```
Default model: `mlx-community/Qwen2.5-3B-Instruct-4bit` (edit in
`agentcore/backends/mlx_backend.py`). First run downloads the weights (~1.9 GB).

## Layout
```
agentcore/
  tool.py        Tool, ToolResult, Sensitivity (safe/confirm/blocked)
  registry.py    tool registry + dependency-free JSON-Schema validation
  parser.py      robust single-JSON-object extraction from model output
  loop.py        the agent loop: generate→parse→validate→confirm→execute→observe
  backends/
    base.py      ModelBackend interface + GenerationConfig
    mock.py      scripted / policy backend (runs anywhere)
    mlx_backend.py  real mlx-lm inference (lazy import; Mac only)
tools/
  mock_device.py in-memory reminders / timer / messages / notes tools
eval/
  tasks.py       task suite + success checks
  runner.py      runs tasks, reports success rate / validity / avg steps
  reference_policy.py  rule-based stand-in "model" (test fixture, not product)
demo.py          CLI demo
tests/           unit tests
```

## What maps to the app
`tool.py`, `registry.py`, `parser.py`, `loop.py`, and `backends/base.py` are the modules that
port 1:1 to Swift `AgentCore`. The prompt template in `../shared/prompts/system.md` and the
tool JSON schemas are shared data — the Swift app loads the same files.
