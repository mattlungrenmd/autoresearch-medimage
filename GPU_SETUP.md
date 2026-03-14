# Compute Setup Guide

This repo is meant to stay usable across a range of hardware instead of one
specific GPU.

## Supported Compute Targets

- NVIDIA GPUs through PyTorch `cuda`
- Apple Silicon through PyTorch `mps`
- CPU-only environments for smoke tests and debugging
- hosted notebook GPUs such as Colab and Kaggle

`train.py` auto-selects `cuda`, then `mps`, then CPU. The queue helper and
recipes are written to be portable across those environments, with batch-size
choices tuned to stay reasonable on modest GPUs.

## Installation Approach

Start with the normal repo install:

```bash
uv sync
```

If your machine needs a platform-specific PyTorch build, install `torch` and
`torchvision` for your system using the official PyTorch instructions, then run
the repo normally with `uv run ...`.

## Local Compute

Typical local workflow:

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
```

For the backbone-first overnight lane on local compute, use:

```bash
uv run python tools/run_recipe_queue.py --preset chestxray14_local_overnight --dataset chestxray14 --session your_session --finalize
```

See [docs/LOCAL_OVERNIGHT_RUN.md](docs/LOCAL_OVERNIGHT_RUN.md) for the full
local workflow.

## Colab

Colab is a good fit for short frontier sweeps or overnight notebook runs.

Typical flow:

```python
!git clone https://github.com/<your-username>/autoresearch-medimage
%cd autoresearch-medimage
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv sync
!uv run prepare.py --dataset chestxray14
!uv run train.py --recipe chexnet --save-checkpoint
```

For the backbone frontier queue, use the Colab runbook in
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md).

## Kaggle And Other Hosted GPUs

Kaggle, Lightning, and cloud VMs can usually run the repo the same way as a
local Linux machine:

```bash
uv sync
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
```

If you are using a notebook service with pre-mounted datasets, adapt only the
dataset path and runtime setup. The training entrypoints stay the same.

## Practical Expectations

- CPU is fine for toy smoke tests and debugging, but not ideal for full
  ChestX-ray14 search loops.
- Consumer GPUs and notebook GPUs are enough for the default 300-second
  research contract.
- Larger GPUs mainly buy you more headroom for heavier backbones, larger image
  sizes, and less aggressive batch-size tuning.

## Overnight Agent Runs

The repo includes a queue helper for repeatable session runs:

```bash
uv run python tools/run_recipe_queue.py --list-presets
```

That helper is an outer-loop convenience. The research contract is still:

- `prepare.py` stays fixed
- `train.py` is the mutation surface
- each experiment uses the fixed 300-second budget by default
- `results.tsv` remains the ledger of keep/discard decisions
