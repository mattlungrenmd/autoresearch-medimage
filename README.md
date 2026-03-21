# autoresearch-medimage

`autoresearch-medimage` is a medical-imaging adaptation of
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

It is built for researchers with labeled 2D medical imaging datasets who want
an agent-first loop for candidate discovery: backbones, losses, augmentations,
and training settings under a fixed experiment contract.

Built by [Matthew Lungren](https://github.com/mattlungrenmd) as a practical
medical-imaging extension of the `autoresearch` idea.

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

This repo stays close to the small-contract spirit of
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch), but it
adapts that workflow to medical imaging.

What stayed the same:

- `prepare.py` stays fixed during experiment loops
- `train.py` remains the main experiment surface
- short runs are compared under a fixed wall-clock budget
- `results.tsv` is still the experiment ledger

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
- `train.py`: the experiment surface for the inner loop
- `program.md`: agent instructions for the search loop
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
train.py    - experiment surface
program.md  - agent instructions
```

Search-time selection should happen on `val`. Use held-out `test` only for the
final report of the winning checkpoint or finalists.

See [program.md](program.md) for the full agent workflow and
[examples/chestxray14/overnight_run.md](examples/chestxray14/overnight_run.md) for a
ChestXray14-specific overnight example runbook.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Acknowledgments

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
the idea that a small, fixed contract can support an autonomous research loop.

## License

MIT. See [LICENSE](LICENSE).
