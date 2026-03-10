# Using With Claude or Codex

This repo uses fast, fixed-budget experiments to search for the best model
configuration for 2D medical imaging tasks. The agent iterates, the winning
config informs a full-scale training run.

## Core Pattern

1. Prepare or validate the dataset.
2. Let the agent search: fast experiments to find the best backbone + hyperparameters.
3. Keep or discard experiments based on the metric in `run.log` and
   `results.tsv`.
4. Train the winning config to convergence with a longer `--time-budget`.

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

- `prepare.py` is fixed during the search loop
- `train.py` is the mutation surface
- `results.tsv` is the untracked experiment ledger
- each search experiment uses the fixed 300-second budget from `prepare.py`
- `--time-budget` is available for quick smoke tests and for the final full training run

## Good Prompts

- "Read the repo, validate my dataset contract, then run a baseline."
- "Use the current best run as the baseline and improve mean AUC without changing `prepare.py`."
- "Generate a draft paper and regulatory planning documents from the best checkpoint."

## What The Repo Does Well Today

- automated architecture and hyperparameter search via fast experiment loops
- built-in recipes for a few public medical imaging benchmarks
- bring-your-own-data for standard 2D tasks
- checkpoint metadata and experiment logging
- paper and regulatory document generation from final model

## What To Treat As Experimental

- foundation-model recipes for external HuggingFace checkpoints
