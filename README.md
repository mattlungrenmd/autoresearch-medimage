# autoresearch-medimage

You have medical images. This repo runs an autoresearch loop to find the best
model and hyperparameters for your task.

An AI agent runs fast, fixed-budget experiments, compares backbones,
hyperparameters, and augmentation strategies, keeps a small frontier of strong
candidates, and keeps mutating that frontier until it stalls.

```
1. Search   - agent runs fixed-budget medical-imaging experiments (5 min each)
2. Evaluate - best kept checkpoint gets a held-out test eval and progress plot
3. Scale    - winning configuration trains to convergence; paper helpers are optional
```

## Quick Start: Your Dataset

```bash
# Auto-detect task type, classes, and splits from any image directory
uv run prepare.py --data /path/to/your/images

# Run a search experiment
uv run train.py --dataset your_dataset
```

Or use the manual contract:

1. Create `data/<dataset>/dataset.yaml` and `data/<dataset>/splits.csv`
2. Run `uv run prepare.py --dataset your_dataset`
3. Run `uv run train.py --dataset your_dataset`

See [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) for details.

## Quick Start: Built-in Demo (CheXNet)

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
```

For a pure autoresearch flow, stop at the progress plot plus held-out eval of
the winning checkpoint. `tools/finalize_run.py` and the paper/regulatory
generators remain optional experimental helpers. See
[docs/CHEXNET_HELLO_WORLD.md](docs/CHEXNET_HELLO_WORLD.md) for the full walkthrough.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlungrengmail/autoresearch-medimage/blob/main/notebooks/colab_quickstart.ipynb)

Try the public Colab paths on a free T4 GPU:

- [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb) for the
  smoke test and bring-your-own-data flow
- [notebooks/colab_chexnet.ipynb](notebooks/colab_chexnet.ipynb) for the
  ChestX-ray14 single-baseline `chexnet` demo using a local folder or archive
  you already have
- [notebooks/colab_autoresearch.ipynb](notebooks/colab_autoresearch.ipynb) for
  a named ChestX-ray14 autoresearch session that seeds a baseline queue, keeps
  a frontier of the best kept runs across multiple backbones, mutates that
  frontier with fixed 300-second experiments, and finishes with a held-out
  eval plus core artifact backup

All three notebooks pull a public repo snapshot into `/content`. No local
setup, GitHub token, or private-repo access is required once the repo is
public.

## Smoke Test (no GPU, no download)

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

## Built-In Datasets

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
uv run prepare.py --list          # list all datasets
uv run train.py --list-recipes    # list all recipes
```

## The Three Files

```
prepare.py  - dataset download, data loading, evaluation (READ-ONLY during search)
train.py    - the experiment surface (agent modifies this)
program.md  - agent instructions for the search loop
```

The agent modifies only `train.py` during the search phase. Once search
converges, the core loop is: inspect `results.tsv`, refresh `progress.png`,
and evaluate the best kept checkpoint on held-out `test`. The paper and
regulatory generators are optional experimental helpers, not the core
autoresearch contract.

See [program.md](program.md) for the full agent workflow.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full plan.

## Acknowledgments

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -
the idea that a small, fixed contract (`prepare.py` + `train.py` + `program.md`)
is enough for an AI agent to run an autonomous research loop.

## License

MIT. See [LICENSE](LICENSE).
