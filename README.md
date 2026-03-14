# autoresearch-medvision

Agent-first autoresearch for 2D medical imaging.

This repo adapts the spirit of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
to short-horizon medical imaging experiments: fixed-budget runs, an untracked
experiment ledger, and `train.py` as the main mutation surface for Claude/Codex.

CheXNet is the canonical hello-world and end-to-end demo, but the repo is
meant to be useful for your own dataset and objective as well.

## What This Repo Is

- A framework for 2D medical imaging experiments on built-in and local datasets
- A simple bring-your-own-data contract under `data/<dataset>/`
- A place for agents to iterate on backbones, losses, and optimization in `train.py`
- An optional AdaEvolve-lite overnight search mode that stays on top of `train.py`
- An experimental home for paper and regulatory planning scaffolds

## Capabilities

| Status | What it means |
|--------|----------------|
| Today | 2D classification, multilabel classification, and binary segmentation |
| Experimental | HuggingFace medical backbones, paper scaffold generation, regulatory scaffolds |
| Planned | MONAI backend, DICOM/NIfTI workflows, 2.5D and 3D support |

## Quick Start

### Bring your own dataset

1. Create `data/<dataset>/dataset.yaml`
2. Create `data/<dataset>/splits.csv`
3. Validate it:

```bash
uv run prepare.py --dataset your_dataset
```

4. Train it:

```bash
uv run train.py --dataset your_dataset
```

See [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) for the full contract.

For a zero-setup local smoke test, the repo also ships a tiny fixture:

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

### Built-in demo: CheXNet

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
uv run python tools/finalize_run.py --dataset chestxray14
```

Built-in benchmark downloads that come from Kaggle, including `chestxray14`,
require a valid Kaggle API token in `~/.kaggle/kaggle.json` or the equivalent
environment variables.

See [docs/CHEXNET_HELLO_WORLD.md](docs/CHEXNET_HELLO_WORLD.md) for the full
hello-world flow, including paper and regulatory scaffold generation.
For the overnight agent workflow, see [docs/OVERNIGHT_CHEXNET_RUN.md](docs/OVERNIGHT_CHEXNET_RUN.md).
For a one-command local compute backbone-first run, see [docs/LOCAL_OVERNIGHT_RUN.md](docs/LOCAL_OVERNIGHT_RUN.md).

The CheXNet demo now uses a stricter `train`/`val` search split plus a separate
held-out `test` evaluation during finalization. That paper-grade reporting path
is specific to benchmark datasets that expose a true final split; the generic
bring-your-own-data contract still works for any dataset with a valid
`splits.csv`.

## Hardware Notes

The repo is meant to stay usable across:

- local NVIDIA GPUs
- Apple Silicon through PyTorch `mps`
- CPU-only setups for smoke tests and debugging
- hosted notebook GPUs such as Colab and Kaggle

`train.py` auto-selects `cuda`, then `mps`, then CPU. The default `uv sync`
install is now hardware-agnostic. If you want a platform-specific PyTorch build,
install `torch` and `torchvision` for your system using the official PyTorch
instructions, then run the repo normally.

For the smoothest fresh install, use Python `3.12`. The repo now pins that via
`.python-version`, and the supported range in `pyproject.toml` excludes Python
`3.14+` because some imaging dependencies do not yet ship friction-free wheels
there.

See [GPU_SETUP.md](GPU_SETUP.md) for practical environment guidance and
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md) for the Colab lane.

## Built-In Datasets

Built-in examples are included for:

- `chestxray14`
- `aptos`
- `oct`
- `chest_xray`
- `brain_mri`
- `retina_vessel`
- `pneumothorax`

List them at any time with:

```bash
uv run prepare.py --list
uv run train.py --list-recipes
```

## Recipes

Recipes are examples and baselines, not the whole product story.

- `chexnet` for the hello-world end-to-end demo
- ophthalmology examples such as `aptos_winner`
- foundation-model examples such as `chexnet_biomedclip`
- generic starter templates for classification, multilabel, and segmentation

## Agent Workflow

The intended inner loop is:

1. validate or download the dataset
2. establish a baseline run
3. let the agent modify `train.py`
4. read the metric in `run.log` and `results.tsv`
5. keep or discard the experiment externally

See [AGENTS.md](AGENTS.md), [program.md](program.md), and
[docs/USING_WITH_CLAUDE_CODEX.md](docs/USING_WITH_CLAUDE_CODEX.md).

## Optional Adaptive Search

The default inner loop is still manual/autoresearch-style. If you want a
session-local adaptive outer loop instead of a fixed queue, the repo now ships
an optional AdaEvolve-lite helper:

```bash
uv run python tools/run_search_session.py \
  --space chestxray14 \
  --session chestxray14_adaevolve_20260314
```

This mode:

- keeps `prepare.py` fixed
- still runs `train.py` for every scored experiment
- writes generated overlay recipes under `logs/<session>/search/recipes/`
- stores search state, lineage, and reports under `logs/<session>/search/`
- records the same run outcomes in `results.tsv`

Every generated candidate is just a recipe overlay, so you can replay any run
directly with:

```bash
uv run train.py --recipe-file logs/<session>/search/recipes/<candidate>.yaml
```

## Experimental Outputs

These tools exist, but they are not the main claim of the repo:

- `uv run paper/generate_paper.py`
- `uv run regulatory/generate_regulatory.py`

Both are experimental scaffolds built from run artifacts. They are not
submission-ready outputs.

To visualize an overnight run ledger, you can also generate a Karpathy-style
progress chart:

```bash
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
```

## Roadmap

The roadmap lives in [ROADMAP.md](ROADMAP.md). Public v1 is intentionally scoped
to 2D tasks and agent workflows. MONAI, multislice, and volumetric support are
planned work, not current functionality.

## License

MIT. See [LICENSE](LICENSE).
