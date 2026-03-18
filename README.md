# autoresearch-medimage

`autoresearch-medimage` is a medical-imaging adaptation of
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

It is built for researchers with labeled 2D medical imaging datasets who want
an agent-first loop for candidate discovery: backbones, losses, augmentations,
and training settings under a fixed experiment contract.

The repo is strongest today for:

- candidate discovery on labeled medical imaging datasets
- reproducible fixed-budget search runs on local GPU or Colab
- staged validation after search, using held-out `test` only at the end

It is not positioned as a turnkey final-model selector. Treat the search loop
as a way to surface strong candidates quickly, then validate the finalists
carefully.

## Current Scope

Public v1 supports:

- 2D classification
- 2D multilabel classification
- binary segmentation

Representative use cases:

- chest X-ray multi-label classification
- fundus or OCT classification
- binary mask segmentation

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

## Quick Start: Built-In Demo (CheXNet)

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
uv run python tools/finalize_run.py --dataset chestxray14
```

See [docs/CHEXNET_HELLO_WORLD.md](docs/CHEXNET_HELLO_WORLD.md) for the
end-to-end walkthrough and the benchmark context against the original CheXNet
paper tables.

## Google Colab

You can use the public notebooks on a free T4 GPU with no local install.

- Quick start:
  [Open in Colab](https://colab.research.google.com/github/mlungrengmail/autoresearch-medimage/blob/main/notebooks/colab_quickstart.ipynb)
- CheXNet hello-world:
  [Open in Colab](https://colab.research.google.com/github/mlungrengmail/autoresearch-medimage/blob/main/notebooks/colab_chexnet.ipynb)
- Multi-run ChestXray14 search:
  [Open in Colab](https://colab.research.google.com/github/mlungrengmail/autoresearch-medimage/blob/main/notebooks/colab_autoresearch.ipynb)

Notebook files in the repo:

- [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb)
- [notebooks/colab_chexnet.ipynb](notebooks/colab_chexnet.ipynb)
- [notebooks/colab_autoresearch.ipynb](notebooks/colab_autoresearch.ipynb)

For the full runbook versions, see [docs/GPU_SETUP.md](docs/GPU_SETUP.md) and
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md).

## Smoke Test

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

## Built-In Datasets

These are built-in presets in `prepare.py`, so you can run them without
creating your own `data/<dataset>/dataset.yaml` first.

| Dataset | Task | Metric | Source |
|---------|------|--------|--------|
| `chestxray14` | Multi-label (14 pathologies) | Mean AUC | NIH ChestX-ray14 |
| `aptos` | Classification (5 grades) | Quadratic Kappa | APTOS 2019 |
| `oct` | Classification (4 classes) | Accuracy | Kermany OCT |
| `chest_xray` | Binary classification | AUC | Chest X-ray Pneumonia |
| `brain_mri` | Segmentation | Dice | LGG MRI |
| `retina_vessel` | Segmentation | Dice | Retinal Blood Vessel |
| `pneumothorax` | Segmentation | Dice | SIIM-ACR |

```bash
uv run prepare.py --list
uv run train.py --list-recipes
```

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

This mode still scores every experiment through `train.py`, but it also writes:

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

See [docs/LOCAL_OVERNIGHT_RUN.md](docs/LOCAL_OVERNIGHT_RUN.md),
[docs/COLAB_OVERNIGHT_RUN.md](docs/COLAB_OVERNIGHT_RUN.md), and
[docs/MEDICAL_IMAGING_AUTORESEARCH.md](docs/MEDICAL_IMAGING_AUTORESEARCH.md)
for the longer-form runbooks and case study.

## Repo Layout

- `prepare.py`: dataset preparation, loaders, and evaluation
- `train.py`: the experiment surface for the inner loop
- `program.md`: agent instructions for the search loop
- `data/`: built-in demos plus custom local datasets
- `recipes/`: reproducible recipe files and baselines
- `search_spaces/`: optional adaptive-search search spaces
- `benchmarks/`: benchmark tables, including CheXNet paper comparisons
- `docs/`: bring-your-own-data, GPU setup, and walkthrough docs
- `notebooks/`: Colab entry points
- `tools/`: progress plots, finalize helpers, and search utilities

## Core Contract

The Karpathy-style inner loop stays small:

```text
prepare.py  - fixed during search
train.py    - experiment surface
program.md  - agent instructions
```

Search-time selection should happen on `val`. Use held-out `test` only for the
final report of the winning checkpoint or finalists.

See [program.md](program.md) for the full agent workflow and
[docs/OVERNIGHT_CHEXNET_RUN.md](docs/OVERNIGHT_CHEXNET_RUN.md) for the
CheXNet-style overnight runbook.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Acknowledgments

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
the idea that a small, fixed contract can support an autonomous research loop.

## License

MIT. See [LICENSE](LICENSE).
