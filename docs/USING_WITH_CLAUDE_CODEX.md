# Using With Claude or Codex

This repo is meant to feel like Karpathy's `autoresearch`, but for 2D medical
imaging tasks.

## Core Pattern

1. Prepare or validate the dataset.
2. Establish a baseline run.
3. Let the agent iterate in `train.py`.
4. Keep or discard experiments based on the metric in `run.log` and
   `results.tsv`.

## Commands

Built-in hello-world:

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
```

The full walkthrough lives in [CHEXNET_HELLO_WORLD.md](CHEXNET_HELLO_WORLD.md).
The overnight runbook lives in [OVERNIGHT_CHEXNET_RUN.md](OVERNIGHT_CHEXNET_RUN.md).

Custom dataset:

```bash
uv run prepare.py --dataset your_dataset
uv run train.py --dataset your_dataset
```

Built-in smoke fixture:

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

## Agent Workflow

- `prepare.py` is fixed during the research loop
- `train.py` is the mutation surface
- `results.tsv` is the untracked experiment ledger
- each run uses the fixed 300-second budget from `prepare.py`
- `--time-budget` is available for quick smoke tests, but the canonical loop stays at 300 seconds

## Good Prompts

- "Read the repo, validate my dataset contract, then run a baseline."
- "Use the current best run as the baseline and improve mean AUC without changing `prepare.py`."
- "Generate a draft paper and regulatory planning documents from the best checkpoint."

## What The Repo Does Well Today

- short-horizon baseline and improvement loops
- built-in recipes for a few public medical imaging benchmarks
- bring-your-own-data for standard 2D tasks
- checkpoint metadata and experiment logging

## What To Treat As Experimental

- foundation-model recipes for external HuggingFace checkpoints
