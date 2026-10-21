# autoresearch-medimage

[![Code checks](https://github.com/mattlungrenmd/autoresearch-medimage/actions/workflows/ci.yml/badge.svg)](https://github.com/mattlungrenmd/autoresearch-medimage/actions/workflows/ci.yml)

**Reproducible candidate discovery for medical imaging AI.**

A Python/PyTorch adaptation of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch)
for labeled 2D medical images. Compare model backbones, losses, augmentations,
and training settings under a fixed experiment budget, then train and evaluate
promising candidates for longer.

Built by [Matthew Lungren](https://github.com/mattlungrenmd).

[Try in Colab](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/notebooks/colab_quickstart.ipynb)
· [Bring your own data](docs/BRING_YOUR_OWN_DATA.md)
· [ChestXray14 case study](examples/chestxray14/case_study.md)

## Project at a Glance

| Area | Implementation |
| --- | --- |
| Tasks | 2D classification, multilabel classification, and binary segmentation |
| Experiments | Fixed wall-clock budgets, reproducible recipes, and a `results.tsv` ledger |
| Search | Optional adaptive search with candidate lineage, refinement, and promotion runs |
| Evaluation | Validation-based candidate selection; held-out test evaluation for final reporting |
| Runtime | Local GPU workflows and Google Colab notebooks |

## Explore the Implementation

| Start here | What to review |
| --- | --- |
| [Dataset contract](docs/BRING_YOUR_OWN_DATA.md) and [`prepare.py`](prepare.py) | Image manifests, splits, labels, masks, and dataset validation |
| [`train.py`](train.py) and [recipe templates](recipes/) | Training, task-specific metrics, and reproducible configuration |
| [Search helper](tools/run_search_session.py) and [tests](tests/) | Experiment orchestration, lineage, and regression coverage |
| [ChestXray14 case study](examples/chestxray14/case_study.md) | Why candidate rankings can change after longer training |

The search loop is intended to discover candidates, with staged validation for
the finalists. The case study documents a local experiment series where
short-run rankings changed after longer training; those observations are not a
claim of clinical validation or general benchmark superiority.

## Current Scope

Public v1 supports 2D classification, 2D multilabel classification, and binary
segmentation. Representative use cases include chest X-ray multilabel
classification, fundus or OCT classification, and binary mask segmentation.

This is research tooling. Current scope does not include 3D, volumetric, or
DICOM-series workflows, and the repository does not establish readiness for
clinical use.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). See
[GPU setup](docs/GPU_SETUP.md) for hardware and runtime guidance, or use the
Colab quick start above.

```bash
git clone https://github.com/mattlungrenmd/autoresearch-medimage.git
cd autoresearch-medimage
uv sync
```

## Quick Start: Your Dataset

```bash
# Auto-detect task type, classes, and splits from a local image folder
uv run prepare.py --data /path/to/your/images

# Run a fixed-budget experiment
uv run train.py --dataset your_dataset
```

Or use the manual contract:

1. Create `data/<dataset>/dataset.yaml` and `data/<dataset>/splits.csv`
2. Run `uv run prepare.py --dataset your_dataset`
3. Run `uv run train.py --dataset your_dataset`

See [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) for the schema.

## Quick Start: Built-In Benchmark Demo (ChestXray14)

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chestxray14_baseline --save-checkpoint
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
uv run python tools/finalize_run.py --dataset chestxray14
```

See [examples/chestxray14/README.md](examples/chestxray14/README.md) for the
end-to-end ChestXray14 walkthrough and the benchmark context against the
original paper tables.

## Google Colab

You can use the public notebooks on a free T4 GPU with no local install.

- Quick start:
  [Open in Colab](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/notebooks/colab_quickstart.ipynb)
- Single benchmark demo:
  [Open in Colab](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/examples/chestxray14/colab_demo.ipynb)
- Multi-run ChestXray14 search:
  [Open in Colab](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/notebooks/colab_autoresearch.ipynb)

Notebook files in the repo:

- [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb)
- [examples/chestxray14/colab_demo.ipynb](examples/chestxray14/colab_demo.ipynb)
- [notebooks/colab_autoresearch.ipynb](notebooks/colab_autoresearch.ipynb)

For the full runbook versions, see [docs/GPU_SETUP.md](docs/GPU_SETUP.md) and
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md).

## Smoke Test

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

## Dataset Prep Wiring

`prepare.py` includes wiring for a handful of public medical-imaging datasets,
including Kaggle-hosted mirrors, but the repo does not try to choose the
benchmark for you.

Use the built-in wiring when it matches your problem, or point the repo at your
own local dataset with `--data`.

```bash
uv run prepare.py --list
uv run train.py --list-recipes
```

## What Changed From `autoresearch`

This repo stays close to the minimal design of
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch), but it
adapts that workflow to medical imaging.

What stayed the same:

- `prepare.py` stays fixed during experiment loops
- `train.py` remains the main training file
- short runs are compared under a fixed wall-clock budget
- `results.tsv` is still the results log

What changed for medical imaging:

- the objective is task-specific (`mean_auc`, F1, Dice, QWK, accuracy), not bits per
  byte
- the dataset contract has to handle labeled image folders, splits, masks, and
  Kaggle-style prep flows
- recipes need to span very different backbone families and image sizes under
  the same time budget
- the repo ships optional outer-loop helpers for branch management, refinement,
  and longer validation because slower image models can rerank later

## Optional Adaptive Search

The smallest loop is still the Karpathy-style contract: run `train.py`, inspect
`results.tsv`, and keep or discard ideas externally. If you want a session-local
adaptive outer loop on top of that contract, the repo also ships an optional
search helper:

```bash
uv run python tools/run_search_session.py \
  --space chestxray14 \
  --session chestxray14_stage1
```

This mode still scores every experiment through `train.py`. The outer loop
mainly adds search policy around that fixed inner loop:

- seed from a search space or prior winner sessions
- keep multiple promising branches alive instead of tunneling on one recipe
- mutate one family at a time across backbones, loss, optimization,
  augmentation, image size, and regularization
- confirm borderline gains before treating them as real improvements
- save lineage, generated overlay recipes, and session state for later review

It writes:

- generated overlay recipes under `logs/<session>/search/recipes/`
- search state and lineage under `logs/<session>/search/`
- the same run outcomes back into `results.tsv`

Built-in search spaces live under `search_spaces/`. The first included example
is [search_spaces/chestxray14.yaml](search_spaces/chestxray14.yaml).

## Optional Staged Validation

For benchmark-heavy work, the repo also supports a staged pattern:

1. broad fixed-budget search
2. refinement search around the top diverse candidates
3. longer promotion runs for the finalists

Build a refinement space from one stage-1 session:

```bash
uv run python tools/build_refinement_space.py \
  --session chestxray14_stage1 \
  --space chestxray14 \
  --top-k 5 \
  --diverse-by branch \
  --output logs/chestxray14_stage1_refine/refinement_space.yaml
```

Or seed a refinement space directly from winner sessions:

```bash
uv run python tools/build_refinement_space.py \
  --seed-session cand0056_fulltrain_ep10 \
  --seed-session cand0026_fulltrain_ep10 \
  --seed-session cand0003_fulltrain_ep10 \
  --space chestxray14 \
  --output logs/chestxray14_stage2_refine/refinement_space.yaml
```

Promote refined finalists with:

```bash
uv run python tools/run_full_train_queue.py \
  --session chestxray14_stage2_refine \
  --top-k 3 \
  --diverse-by branch \
  --epochs 10
```

Or validate one candidate directly:

```bash
uv run python tools/run_full_train.py \
  --session chestxray14_stage2_refine \
  --epochs 10 \
  --train-session chestxray14_stage2_refine_fulltrain
```

See [docs/LOCAL_OVERNIGHT_RUN.md](docs/LOCAL_OVERNIGHT_RUN.md) and
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md) for the longer-form
runbooks. A separate ChestXray14 case study lives in
[examples/chestxray14/case_study.md](examples/chestxray14/case_study.md).

## Repo Layout

- `prepare.py`: dataset preparation, loaders, and evaluation
- `train.py`: the training file used by each run
- `program.md`: optional prompt/instruction file for Claude or Codex sessions
- `data/`: built-in demos plus custom local datasets
- `recipes/`: reproducible recipe files and baselines
- `search_spaces/`: optional adaptive-search search spaces
- `benchmarks/`: benchmark tables and reference comparisons
- `examples/`: contained benchmark-specific example assets
- `docs/`: bring-your-own-data, GPU setup, and walkthrough docs
- `notebooks/`: Colab entry points
- `tools/`: progress plots, finalize helpers, and search utilities

## Core Contract

The Karpathy-style inner loop stays small:

```text
prepare.py  - fixed during search
train.py    - training file
program.md  - optional instructions for Claude/Codex sessions
```

Search-time selection should happen on `val`. Use held-out `test` only for the
final report of the winning checkpoint or finalists.

See [program.md](program.md) for the full Claude/Codex workflow and
[examples/chestxray14/overnight_run.md](examples/chestxray14/overnight_run.md) for a
ChestXray14-specific overnight example runbook.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Development

Install the locked runtime and development dependencies, then run the static
checks and CPU regression suite:

```bash
uv sync --locked
uv run ruff check .
uv run python -m unittest discover -s tests -v
```

The suite covers task metrics, result logging, candidate selection, recipe
promotion, search orchestration, and report generation. GitHub Actions runs the
same checks on macOS with Python 3.12. GPU performance and benchmark replication
are separate from these checks.

## About the Author

[Matthew P. Lungren, MD, MPH](https://profiles.stanford.edu/matthew-lungren) is a
physician-scientist focused on clinical AI evaluation and multimodal medical AI,
co-founder of Stanford AIMI, and former Chief Scientific Officer for Health and
Life Sciences at Microsoft.

Selected collaborative work:

- [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays): chest
  radiograph data and evaluation.
- [MedHELM](https://doi.org/10.1038/s41591-025-04151-2): holistic evaluation of
  language models for medical tasks.
- [Healthcare Agent Orchestrator](https://github.com/Azure-Samples/healthcare-agent-orchestrator):
  modular agents for multidisciplinary healthcare workflows.

## Acknowledgments

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
the idea that a small, fixed contract can support an autonomous research loop.

## License

MIT. See [LICENSE](LICENSE).
