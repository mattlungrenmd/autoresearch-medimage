# Overnight CheXNet Run

This is the runbook for a faithful Karpathy-style overnight session in this
repo.

## Goal

Start a fresh named session, let the agent iterate on `train.py` for 6-8
hours, and end with:

- an updated `results.tsv`
- a `progress.png` plot
- an updated paper scaffold with a held-out `test` result
- updated regulatory planning scaffolds

## Preconditions

- `data/chestxray14/` is already prepared
- `data/chestxray14/splits.csv` contains `train`, `val`, and `test`
- the workspace has a clean session tag ready (for example `chexnet_overnight_20260310`)
- `prepare.py` stays fixed during the run
- `train.py` is the only experiment surface

For safety, run the overnight session in an isolated workspace or clone so the
main repo can still be edited separately without affecting the experiment.

## Session Baseline

Do not use literature numbers as the keep/discard baseline.

Instead:
- choose a new session tag
- run the first experiment in that session
- treat that first completed experiment as the session baseline
- compare every later run only to the current best kept run from that same session

Example:

```bash
uv run train.py --recipe chexnet --session chexnet_overnight_20260310 --save-checkpoint > run.log 2>&1
uv run python tools/mark_result.py --dataset chestxray14 --session chexnet_overnight_20260310 --status keep --append-note "session baseline"
```

The recommended evaluation protocol is the paper-grade split protocol:
- search and select on `val`
- evaluate the best kept checkpoint on held-out `test` only during finalization

## Kickoff Prompt

Use this as the starting prompt for Claude or Codex:

```text
Read README.md, CLAUDE.md, AGENTS.md, prepare.py, train.py, program.md, docs/CHEXNET_HELLO_WORLD.md, docs/OVERNIGHT_CHEXNET_RUN.md, and results.tsv.

We are running an overnight Karpathy-style autoresearch session on chestxray14 in this isolated workspace.

Session setup:
- dataset: chestxray14
- session tag: chexnet_overnight_20260310
- selection metric: validation mean_auc
- the first completed experiment in this session becomes the session baseline
- held-out test is reserved for finalization only

Mission:
Improve the kept chestxray14 validation mean_auc over an overnight run on available local compute or a hosted notebook GPU, while staying faithful to the autoresearch workflow and treating the result as a candidate-discovery stage rather than an automatic final ranking.

Hard rules:
- Do not modify prepare.py.
- Only modify train.py during the experiment loop.
- Pass --session chexnet_overnight_20260310 on every training run.
- Do not use the held-out test split to rank experiments.
- Do not use literature numbers for keep/discard decisions.
- Use validation mean_auc only for keep/discard decisions during the loop.
- Use the default 300-second training budget per run.
- Redirect each run to run.log.
- After each run, inspect run.log and results.tsv.
- Mark the latest row as keep or discard with tools/mark_result.py using the same session tag.
- Every completed experiment must end with an explicit keep/discard decision.
- Prefer simple, high-yield changes over complicated rewrites.
- Do not make unsupported scientific claims.
- Do not stop after one experiment.

Search strategy:
- Start with architecture/backbone changes and obvious training improvements.
- Then tune optimizer, LR, weight decay, dropout, loss, batch size, and augmentation.
- Favor changes likely to fit comfortably on modest local GPUs and common notebook GPUs.
- If a change clearly regresses, discard it and move on.
- If a change improves validation mean_auc, keep it and continue from that new session baseline.
- If a change only improves by a tiny amount, prefer a confirmation rerun before treating it as a real gain.

Stop condition:
- Run 50 completed experiments after the session baseline.
- Stop earlier only if I interrupt you or you hit a real blocker.
- Do not send a final summary before 50 completed experiments unless you hit a real blocker.

Progress behavior:
- Keep going without asking for confirmation between experiments unless you hit a real blocker.
- Give short progress updates periodically with:
  - experiment count completed
  - current best validation mean_auc
  - whether the latest run was kept or discarded
  - what changed in train.py

End-of-run steps:
- Run tools/finalize_run.py --dataset chestxray14 --session chexnet_overnight_20260310
- Summarize:
  - number of experiments run
  - number of kept improvements
  - best kept validation mean_auc
  - held-out test mean_auc from finalization
  - best checkpoint path
  - generated progress plot path
  - generated paper path
  - generated figures path
  - generated regulatory output path

Begin now.
```

Important:
- Use a concrete experiment-count stop condition, not only "continue until interrupted."
- Update the session tag in the prompt when you launch a new overnight session.

## During The Run

The core rhythm is:

```bash
uv run train.py --session chexnet_overnight_20260310 > run.log 2>&1
uv run python tools/mark_result.py --dataset chestxray14 --session chexnet_overnight_20260310 --status keep
uv run python tools/mark_result.py --dataset chestxray14 --session chexnet_overnight_20260310 --status discard
```

Which status to use depends on whether the latest run improved on the current
best validation result from the same session.

## End Of Run

Regenerate the run artifacts:

```bash
uv run python tools/finalize_run.py --dataset chestxray14 --session chexnet_overnight_20260310 --compile-paper
```

If `pdflatex` is not available, the paper generator will leave the `.tex` file
in place and continue.

## Outputs

By default the finalize step refreshes:

- `progress.png`
- `paper/generated/chestxray14_final_eval.pt`
- `paper/generated/chestxray14_figures/`
- `paper/generated/chestxray14_overnight.tex`
- `regulatory/generated/chestxray14_overnight/`

Those are the main artifacts to review in the morning.

## Optional Medical Imaging Follow-Up

For a faithful Karpathy-style overnight run, the instructions above are enough.
If you want a more benchmark-oriented medical-imaging workflow, treat that
overnight session as stage 1 rather than the final ranking:

1. run the short-budget overnight search
2. build a refinement search space from the top diverse candidates
3. run a longer-budget refinement session
4. promote the best refined candidates into longer full-train runs

Example:

```bash
uv run python tools/build_refinement_space.py --session chexnet_overnight_20260310 --space chestxray14 --top-k 5 --diverse-by branch --output logs/chexnet_overnight_20260310_refine/refinement_space.yaml
uv run python tools/run_search_session.py --space logs/chexnet_overnight_20260310_refine/refinement_space.yaml --session chexnet_overnight_20260310_refine --time-budget 1800
uv run python tools/run_full_train_queue.py --session chexnet_overnight_20260310_refine --top-k 3 --diverse-by branch --epochs 10
```

If your strongest promoted checkpoints already sit in separate sessions, the
same helper can seed a refinement space directly from those winner sessions:

```bash
uv run python tools/build_refinement_space.py --seed-session cand0056_fulltrain_ep10 --seed-session cand0026_fulltrain_ep10 --seed-session cand0003_fulltrain_ep10 --space chestxray14 --mutation-family lr_wd --mutation-family optimizer_scheduler --mutation-family batch_image --mutation-family regularization --mutation-family augmentation --mutation-family loss --max-runs 24 --output logs/chestxray14_stage2_refine/refinement_space.yaml
```

This repo keeps the short fixed wall-clock loop because it is still a useful
screening contract. The optional follow-up exists for cases where slower
backbone families need more time before the final ranking is trustworthy.
