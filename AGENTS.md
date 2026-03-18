# AGENTS.md

## Purpose

This repo is an agent-first medical imaging research framework inspired by
Karpathy's `autoresearch`.

The public v1 scope is:

- 2D classification
- 2D multilabel classification
- binary segmentation

## Inner Loop Contract

- `prepare.py` is fixed during experiment loops
- `train.py` is the experiment mutation surface
- every experiment uses the fixed 300-second budget from `prepare.py`
- `results.tsv` is the untracked experiment ledger
- `--time-budget` exists for smoke tests, but the default research loop stays at 300s

## Common Commands

```bash
uv run prepare.py --list
uv run prepare.py --dataset your_dataset
uv run train.py --dataset your_dataset
uv run train.py --recipe chexnet --save-checkpoint
uv run train.py --recipe toy_binary_smoke --time-budget 5
uv run train.py --list-recipes
uv run python tools/run_search_session.py --space chestxray14 --session your_session
uv run python tools/build_refinement_space.py --session your_session --space chestxray14 --top-k 5 --diverse-by branch --output logs/your_session_refine/refinement_space.yaml
uv run python tools/build_refinement_space.py --seed-session cand0056_fulltrain_ep10 --seed-session cand0026_fulltrain_ep10 --seed-session cand0003_fulltrain_ep10 --space chestxray14 --mutation-family lr_wd --mutation-family optimizer_scheduler --mutation-family batch_image --mutation-family regularization --mutation-family augmentation --mutation-family loss --max-runs 24 --output logs/chestxray14_stage2_refine/refinement_space.yaml
uv run python tools/run_full_train.py --session your_session --epochs 10
uv run python tools/run_full_train_queue.py --session your_session --top-k 3 --diverse-by branch --epochs 10
uv run python tools/mark_result.py --dataset chestxray14 --status keep
uv run paper/generate_paper.py
uv run regulatory/generate_regulatory.py
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
uv run python tools/finalize_run.py --dataset chestxray14 --compile-paper
```

## Bring Your Own Data

Custom datasets live under `data/<dataset>/` with:

- `dataset.yaml`
- `splits.csv`

See `docs/BRING_YOUR_OWN_DATA.md` for the schema.

## Rules For Agents

- establish a baseline before making changes
- prefer changing `train.py` during research iterations
- use built-in datasets and recipes as examples, not as the only intended workflow
- treat the paper and regulatory generators as experimental helpers
- for medical-imaging benchmarks, treat the short wall-clock search as a screening stage and use promotion/refinement helpers before declaring a final winner
- frame repo outcomes as candidate discovery plus staged validation, not as automatic final-model selection
- do not claim MONAI, 2.5D, or 3D support in the current repo state
