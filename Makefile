# DeviceControlBench — reproducible entry points.
# Quality eval is portable (CPU/Linux/Mac); efficiency eval uses MLX on a Mac.

export PYTHONPATH := $(CURDIR):$(CURDIR)/python

.PHONY: help seed gold test eval eval-mlx summarize

help:
	@echo "make seed       - (re)generate the seed benchmark JSONL"
	@echo "make gold       - gold-replay validation (every task passes its checker)"
	@echo "make test       - agent-core unit tests"
	@echo "make eval MODEL=Qwen/Qwen2.5-1.5B-Instruct   - run benchmark (HF backend, CPU/GPU)"
	@echo "make eval-mlx MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit  - run on Apple Silicon"
	@echo "make summarize  - aggregate results/ into a leaderboard"

seed:
	python3 -m bench.build_seed

gold:
	python3 -m bench.gold_replay

test:
	cd python && python3 -m unittest discover -s tests -q

MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
eval:
	python3 -m bench.harness --backend hf --model $(MODEL) --temperature 0.0 --verbose

eval-mlx:
	python3 -m bench.harness --backend mlx --model $(MODEL) --temperature 0.0 --verbose

summarize:
	python3 -m bench.summarize
