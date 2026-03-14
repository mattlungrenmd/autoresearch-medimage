# Local Compute Overnight Run

This runbook is the simplest way to launch a backbone-first ChestX-ray14 search
on local compute without manually babysitting every keep or discard.

If you want an adaptive outer loop instead of a fixed queue, use
`tools/run_search_session.py` with the built-in `chestxray14` search space.

The workflow is tested on a local RTX 3060, but it is not restricted to that
GPU. Any reasonably supported local setup can use the same queue helper:

- local NVIDIA GPU
- Apple Silicon through PyTorch `mps`
- CPU for smoke tests

## Why This Lane Exists

The short-budget search can easily get trapped mutating the current winner.
That is useful once a strong backbone is already found, but it is not the best
way to answer architecture questions such as:

- should we still try DenseNet again?
- should we re-check ResNet, ConvNeXt, and a slightly larger EfficientNet?

For a fixed 300-second budget, architecture choice is part of the optimization
problem. The local overnight lane starts broad on backbone families and only
then lets the session ledger tell us what deserves more attention.

## New Helper

The repo includes:

```bash
uv run python tools/run_recipe_queue.py --list-presets
```

Two presets matter here:

- `chestxray14_backbone_frontier`
- `chestxray14_local_overnight`

The helper:

- runs each recipe under a shared session tag
- saves a per-run log in `logs/<session>/`
- compares each completed run against the current best metric in that session
- marks the latest row in `results.tsv` as `keep` or `discard`
- can finalize the session at the end

## Optional AdaEvolve-Lite Lane

If you want the repo to keep multiple backbone branches alive and adapt which
mutation families it trusts overnight, launch the optional adaptive mode:

```bash
$env:SESSION = "chestxray14_adaevolve_local_20260314"
uv run python tools/run_search_session.py `
  --space chestxray14 `
  --session $env:SESSION
```

That mode still uses `train.py` for every scored run, but it writes generated
overlay recipes plus lineage/search state under `logs/<session>/search/`.

## Recommended Command

Choose a fresh session tag:

```bash
$env:SESSION = "chestxray14_local_overnight_20260313"
```

Kick off the full local overnight lane:

```bash
uv run python tools/run_recipe_queue.py `
  --preset chestxray14_local_overnight `
  --dataset chestxray14 `
  --session $env:SESSION `
  --finalize
```

If you want the shorter architecture-only sweep first:

```bash
uv run python tools/run_recipe_queue.py `
  --preset chestxray14_backbone_frontier `
  --dataset chestxray14 `
  --session $env:SESSION
```

## What The Overnight Preset Runs

The preset mixes existing DenseNet anchors with new architecture-first recipes:

- `chexnet`
- `chexnet_improved`
- `chexnet_advanced`
- `chestxray14_frontier_densenet121`
- `chestxray14_frontier_densenet169`
- `chestxray14_frontier_resnet50`
- `chestxray14_frontier_convnext_tiny`
- `chestxray14_frontier_efficientnet_b0`
- `chestxray14_frontier_efficientnet_b1`
- stronger-color second passes for each of the six frontier backbones

The frontier recipes keep the training setup mostly fixed and only adjust a few
obvious fit knobs such as batch size so the comparison stays fair on modest
local GPUs while still running on larger cards or hosted notebook GPUs.

## Artifacts

During the run, look in:

- `results.tsv`
- `logs/<session>/`
- `logs/<session>/session_summary.md`

If `--finalize` is enabled, the helper also refreshes:

- `progress.png`
- `paper/generated/chestxray14_final_eval.pt`
- `paper/generated/chestxray14_figures/`
- `paper/generated/chestxray14_overnight.tex`
- `regulatory/generated/chestxray14_overnight/`

## Resume After Interruption

The helper skips recipes that already have a completed non-error row in the same
session. Re-run the same command with the same session tag to resume:

```bash
uv run python tools/run_recipe_queue.py `
  --preset chestxray14_local_overnight `
  --dataset chestxray14 `
  --session $env:SESSION `
  --finalize
```

To force a full rerun of every recipe in the queue:

```bash
uv run python tools/run_recipe_queue.py `
  --preset chestxray14_local_overnight `
  --dataset chestxray14 `
  --session $env:SESSION `
  --no-resume
```

## Practical Notes

- Foundation-model recipes are intentionally left out of this preset because
  first-run downloads and larger memory footprints make the overnight lane less
  stable on consumer GPUs and short-lived notebook runtimes.
- The queue helper is an outer-loop convenience. The inner-loop research
  contract is unchanged: `prepare.py` stays fixed, `train.py` remains the main
  mutation surface, and the default research budget is still 300 seconds.
