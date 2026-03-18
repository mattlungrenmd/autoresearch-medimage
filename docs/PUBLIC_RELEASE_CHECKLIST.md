# Public Release Checklist

Use this checklist before pushing a public launch update.

## Repo Hygiene

- `git status` only shows intentional source/doc/test changes
- no local datasets, checkpoints, logs, or generated benchmark artifacts are staged
- run a cleanup dry-run first:

```bash
uv run python tools/cleanup_artifacts.py \
  --keep-best 3 \
  --keep-log-session your_active_session \
  --remove-smoke-logs \
  --remove-tmp \
  --remove-progress-png
```

- if the plan looks right, rerun with `--apply`

## Smoke Validation

- fresh copy or fresh clone installs cleanly with:

```bash
uv sync
```

- quick-start smoke path works:

```bash
uv run train.py --list-recipes
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

- built-in benchmark docs still match the actual commands in:
  - `README.md`
  - `docs/LOCAL_OVERNIGHT_RUN.md`
  - `docs/COLAB_OVERNIGHT_RUN.md`
  - `docs/OVERNIGHT_CHEXNET_RUN.md`

## Benchmark Artifacts

- keep the final winning checkpoints only
- keep held-out evaluation payloads for the finalists you want to report
- if multiple finalists are being evaluated, use isolated outputs with:

```bash
uv run python tools/finalize_run.py \
  --results results.tsv \
  --dataset chestxray14 \
  --session your_session \
  --final-eval-output logs/release_artifacts/your_session/final_eval.pt \
  --figures-output-dir logs/release_artifacts/your_session/figures
```

- make sure benchmark case-study docs are clearly framed as worked examples, not universal claims

## Messaging

- keep the README default path simple:
  - bring your own data
  - run a short search or baseline
  - optional staged workflow for benchmark-heavy use
- avoid claiming support for features not actually present in the repo today
- avoid turning one benchmark result into a universal medical-imaging claim

## Final Checks

- run the relevant test subset or full suite
- verify `README.md` links resolve
- verify `.gitignore` covers local artifacts you do not want to ship
- confirm the active branch is the one you intend to publish
