# Colab Overnight Run

This runbook sets up a separate CheXNet overnight lane on Google Colab without
mixing it into an existing local session.

## Goal

Run a second agent-driven search on `chestxray14` in Colab with:

- a separate workspace
- a separate session tag
- a separate `results.tsv`
- checkpoints and artifacts copied to Google Drive for recovery
- an optional fixed recipe queue for architecture-first sweeps
- an optional AdaEvolve-lite adaptive search lane

This is meant to complement a local overnight run, not replace it.

## Guardrails

- Keep `prepare.py` fixed during the experiment loop.
- Treat `train.py` as the experiment mutation surface.
- Use the default 300-second budget for the research loop.
- Use a new session tag such as `chexnet_overnight_colab_20260310`.
- Do not share `results.tsv` with a local overnight workspace while both are active.
- Do not use held-out `test` for inner-loop keep/discard decisions.

## Recommended Layout

Use Colab local disk for active work and Google Drive only for persistence:

- working copy: `/content/autoresearch-medimage-colab`
- persisted outputs:
  - `/content/drive/MyDrive/autoresearch-medimage-colab/results/`
  - `/content/drive/MyDrive/autoresearch-medimage-colab/checkpoints/`
  - `/content/drive/MyDrive/autoresearch-medimage-colab/artifacts/`

Do not train directly from Drive if you can avoid it. Copy data and code into
`/content/...` first, then sync outputs back to Drive.

## Bootstrap Notebook Cells

### 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2. Choose a Session Tag

```python
SESSION = "chexnet_overnight_colab_20260310"
WORKDIR = "/content/autoresearch-medimage-colab"
DRIVE_ROOT = "/content/drive/MyDrive/autoresearch-medimage-colab"
```

### 3. Clone the Repo

If GitHub is good enough as the starting point:

```bash
!rm -rf /content/autoresearch-medimage-colab
!git clone https://github.com/mlungrengmail/autoresearch-medimage.git /content/autoresearch-medimage-colab
```

If you want the exact state of a local workspace instead, upload a zip snapshot
to Drive first and unpack that into `/content/autoresearch-medimage-colab`
instead of cloning from GitHub.

### 4. Install Dependencies

```bash
%cd /content/autoresearch-medimage-colab
!python -m pip install -q uv
!uv sync
!nvidia-smi
```

### 5. Configure Kaggle Credentials

Upload `kaggle.json` to Drive and then copy it into place:

```bash
!mkdir -p /root/.kaggle
!cp /content/drive/MyDrive/kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
```

### 6. Prepare the Dataset

```bash
%cd /content/autoresearch-medimage-colab
!uv run prepare.py --dataset chestxray14
```

### 7. Run the Session Baseline

```bash
%cd /content/autoresearch-medimage-colab
!uv run train.py --recipe chexnet --session $SESSION --save-checkpoint > run.log 2>&1
!tail -n 120 run.log
!uv run python tools/mark_result.py --dataset chestxray14 --session $SESSION --status keep --append-note "session baseline"
```

### 7b. Recommended: Run the Backbone Frontier Queue

If the local machine is already running a longer overnight lane, the best use of
Colab is usually a shorter separate frontier session instead of repeating the
same exploit-heavy mutation path.

Suggested Colab session tag:

```python
SESSION = "chestxray14_colab_frontier_20260313"
```

Run the queue helper:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_recipe_queue.py \
  --preset chestxray14_backbone_frontier \
  --dataset chestxray14 \
  --session $SESSION
```

That preset runs:

- `chestxray14_frontier_densenet121`
- `chestxray14_frontier_densenet169`
- `chestxray14_frontier_resnet50`
- `chestxray14_frontier_convnext_tiny`
- `chestxray14_frontier_efficientnet_b0`
- `chestxray14_frontier_efficientnet_b1`

It writes per-run logs to `logs/<session>/` and marks `keep` or `discard`
automatically in `results.tsv`.

### 7c. Optional: Run the Adaptive Search Lane

If you want Colab to stay recipe-first but adapt parent selection and mutation
families instead of walking a fixed queue:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_search_session.py \
  --space chestxray14 \
  --session $SESSION
```

This keeps `train.py` as the execution surface and writes search-sidecar
artifacts under `logs/<session>/search/`.

### 7d. Optional: Stage The Search

If you want a more careful benchmark-style follow-up, a useful pattern is:

1. broad stage-1 search
2. refinement search around the top diverse candidates
3. longer promotion runs for the finalists

If your Colab GPU budget allows it, a stronger stage-1 screen is often worth
trying:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_search_session.py \
  --space chestxray14 \
  --session $SESSION \
  --time-budget 900
```

Then materialize a refinement space from the top diverse stage-1 candidates:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/build_refinement_space.py \
  --session $SESSION \
  --space chestxray14 \
  --top-k 5 \
  --diverse-by branch \
  --output logs/${SESSION}_refine/refinement_space.yaml

!uv run python tools/run_search_session.py \
  --space logs/${SESSION}_refine/refinement_space.yaml \
  --session ${SESSION}_refine \
  --time-budget 1800
```

If your best promoted candidates already live in separate sessions, you can
seed refinement directly from those winner sessions instead:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/build_refinement_space.py \
  --seed-session cand0056_fulltrain_ep10 \
  --seed-session cand0026_fulltrain_ep10 \
  --seed-session cand0003_fulltrain_ep10 \
  --space chestxray14 \
  --mutation-family lr_wd \
  --mutation-family optimizer_scheduler \
  --mutation-family batch_image \
  --mutation-family regularization \
  --mutation-family augmentation \
  --mutation-family loss \
  --max-runs 24 \
  --output logs/chestxray14_stage2_refine/refinement_space.yaml
```

Finally, run longer validation on the refined finalists:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_full_train_queue.py \
  --session ${SESSION}_refine \
  --top-k 3 \
  --diverse-by branch \
  --epochs 10
```

If you are driving the promotion queue from one environment but want the actual
training runs to use a different Python interpreter, pass
`--python-executable /path/to/python`.

If you only want the single best refined candidate instead:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_full_train.py \
  --session ${SESSION}_refine \
  --epochs 10 \
  --train-session ${SESSION}_refine_fulltrain
```

### 8. Sync Artifacts Back to Drive

```bash
!mkdir -p "$DRIVE_ROOT/results" "$DRIVE_ROOT/checkpoints" "$DRIVE_ROOT/artifacts"
!cp /content/autoresearch-medimage-colab/results.tsv "$DRIVE_ROOT/results/results.tsv"
!cp /content/autoresearch-medimage-colab/run.log "$DRIVE_ROOT/artifacts/run.log"
!cp -r /content/autoresearch-medimage-colab/checkpoints/. "$DRIVE_ROOT/checkpoints/" || true
!mkdir -p "$DRIVE_ROOT/logs"
!cp -r /content/autoresearch-medimage-colab/logs/. "$DRIVE_ROOT/logs/" || true
```

## Agent Prompt

Use this in a fresh agent session attached to the Colab workspace:

```text
Read AGENTS.md, docs/OVERNIGHT_CHEXNET_RUN.md, docs/CHEXNET_HELLO_WORLD.md, CLAUDE.md, program.md, prepare.py, train.py, results.tsv, and run.log if present.

We are running a separate Colab overnight session on chestxray14.

Session setup:
- dataset: chestxray14
- session tag: chexnet_overnight_colab_20260310
- selection metric: validation mean_auc
- the first completed experiment in this session is the session baseline
- held-out test is reserved for finalization only

Hard rules:
- Do not modify prepare.py.
- Only modify train.py during the experiment loop.
- Pass --session chexnet_overnight_colab_20260310 on every training run.
- Use the default 300-second training budget per run.
- After each run, inspect run.log and results.tsv.
- Mark every completed experiment as keep or discard with tools/mark_result.py using the same session tag.
- Do not use held-out test to rank experiments.
- Continue one experiment at a time; do not add orchestrators or multi-run helper scripts.

Stop condition:
- Run 50 completed experiments after the session baseline.

Begin by reconstructing the current state of session chexnet_overnight_colab_20260310 from results.tsv and checkpoints, then continue from the safest recoverable point.
```

Replace the session tag in the prompt if you launch on a different date.

## Queue-Driven Alternative

If you do not want the agent to mutate `train.py` mid-session and instead want a
clean architecture-first sweep on Colab, use the queue helper directly:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/run_recipe_queue.py \
  --preset chestxray14_backbone_frontier \
  --dataset chestxray14 \
  --session $SESSION
```

After the frontier finishes, the medical-imaging workflow should usually move
into the staged path above instead of immediately trusting the best short-budget
row as the final winner.

## Recovery After Colab Disconnects

When a runtime disconnects:

1. start a fresh Colab runtime
2. mount Drive
3. restore the repo into `/content/autoresearch-medimage-colab`
4. restore `results.tsv`, `run.log`, and `checkpoints/` from Drive
5. continue the same session tag

Suggested restore cells:

```bash
!rm -rf /content/autoresearch-medimage-colab
!git clone https://github.com/mlungrengmail/autoresearch-medimage.git /content/autoresearch-medimage-colab
%cd /content/autoresearch-medimage-colab
!python -m pip install -q uv
!uv sync
!cp "$DRIVE_ROOT/results/results.tsv" /content/autoresearch-medimage-colab/results.tsv
!cp "$DRIVE_ROOT/artifacts/run.log" /content/autoresearch-medimage-colab/run.log || true
!mkdir -p /content/autoresearch-medimage-colab/checkpoints
!cp -r "$DRIVE_ROOT/checkpoints/." /content/autoresearch-medimage-colab/checkpoints/ || true
!mkdir -p /content/autoresearch-medimage-colab/logs
!cp -r "$DRIVE_ROOT/logs/." /content/autoresearch-medimage-colab/logs/ || true
```

After restore, inspect the session rows in `results.tsv` and continue from the
same session tag instead of creating a fresh session.

## Finalization

When the session is done:

```bash
%cd /content/autoresearch-medimage-colab
!uv run python tools/finalize_run.py --dataset chestxray14 --session $SESSION --compile-paper
!cp /content/autoresearch-medimage-colab/progress.png "$DRIVE_ROOT/artifacts/progress.png" || true
!cp -r /content/autoresearch-medimage-colab/paper/generated "$DRIVE_ROOT/artifacts/paper-generated" || true
!cp -r /content/autoresearch-medimage-colab/regulatory/generated "$DRIVE_ROOT/artifacts/regulatory-generated" || true
!cp -r /content/autoresearch-medimage-colab/logs/. "$DRIVE_ROOT/logs/" || true
```

## Practical Notes

- If you already have an active local overnight session, keep the Colab session
  name distinct so the ledgers stay independent.
- If you need the exact current local code state, upload a zip snapshot of the
  local workspace rather than cloning from GitHub.
- Persist `results.tsv` and checkpoints often enough that a disconnect only
  loses the current in-flight experiment.
