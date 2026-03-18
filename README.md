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

# Run a fixed-budget search experiment
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

## Repo Layout

- `prepare.py`: dataset preparation, loaders, and evaluation
- `train.py`: the experiment surface for the inner loop
- `program.md`: agent instructions for the search loop
- `data/`: built-in demos plus custom local datasets
- `recipes/`: reproducible recipe files and baselines
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
