# Overnight CheXNet Run

This is the runbook for a faithful Karpathy-style overnight session in this
repo.

## Goal

Start from the current CheXNet hello-world baseline, let the agent iterate on
`train.py` for 6-8 hours, and end with:

- an updated `results.tsv`
- a `progress.png` plot
- an updated paper generated from results with a held-out `test` result
- updated regulatory planning documents

## Preconditions

- `data/chestxray14/` is already prepared
- `data/chestxray14/splits.csv` contains `train`, `val`, and `test`
- the baseline run exists in `results.tsv`
- `prepare.py` stays fixed during the run
- `train.py` is the only experiment surface

For safety, run the overnight session in an isolated workspace or clone so the
main repo can still be edited separately without affecting the experiment.

## Baseline

The current baseline is the `chexnet` recipe on `chestxray14`.

Before the overnight loop begins, mark the baseline row as kept:

```bash
uv run python tools/mark_result.py --dataset chestxray14 --status keep --append-note "baseline"
```

The recommended baseline is the paper-grade split protocol:
- search and select on `val`
- evaluate the best kept checkpoint on held-out `test` only during finalization

## Kickoff Prompt

Use this as the starting prompt for Claude or Codex:

```text
Read README.md, CLAUDE.md, AGENTS.md, prepare.py, train.py, program.md, docs/CHEXNET_HELLO_WORLD.md, docs/OVERNIGHT_CHEXNET_RUN.md, and results.tsv.

We are running an overnight Karpathy-style autoresearch session on chestxray14 in this isolated workspace.

Current baseline:
- dataset: chestxray14
- kept baseline: chexnet / densenet121
- selection metric: validation mean_auc
- current kept validation mean_auc: 64.41
- held-out test is reserved for finalization only

Mission:
Improve the kept chestxray14 validation mean_auc over an overnight run on the local RTX 3060, while staying faithful to the autoresearch workflow.

Hard rules:
- Do not modify prepare.py.
- Only modify train.py during the experiment loop.
- Do not use the held-out test split to rank experiments.
- Use validation mean_auc only for keep/discard decisions during the loop.
- Use the default 300-second training budget per run.
- Redirect each run to run.log.
- After each run, inspect run.log and results.tsv.
- Mark the latest row as keep or discard with tools/mark_result.py.
- Prefer simple, high-yield changes over complicated rewrites.
- Do not make unsupported scientific claims.
- Do not stop after one experiment.

Search strategy:
- Start with architecture/backbone changes and obvious training improvements.
- Then tune optimizer, LR, weight decay, dropout, loss, batch size, and augmentation.
- Favor changes likely to fit comfortably on the RTX 3060.
- If a change clearly regresses, discard it and move on.
- If a change improves validation mean_auc, keep it and continue from that new baseline.

Runtime target:
- Continue the loop until 6:30 AM America/Los_Angeles on March 10, 2026, or until I interrupt you, whichever comes first.
- Do not send a final summary before that time unless you hit a real blocker.

Progress behavior:
- Keep going without asking for confirmation between experiments unless you hit a real blocker.
- Give short progress updates periodically with the current best validation mean_auc and what changed.

End-of-run steps:
- Run tools/finalize_run.py --dataset chestxray14
- Summarize:
  - number of experiments run
  - best kept validation mean_auc
  - held-out test mean_auc from finalization
  - best checkpoint path
  - generated paper path
  - generated figures path
  - generated regulatory output path

Begin now.
```

Important:
- Use an absolute local deadline, not only "continue until interrupted."
- Update the date/time in the prompt when you launch a new overnight session.

## During The Run

The core rhythm is:

```bash
uv run train.py > run.log 2>&1
uv run python tools/mark_result.py --dataset chestxray14 --status keep
uv run python tools/mark_result.py --dataset chestxray14 --status discard
```

Which status to use depends on whether the latest run improved on the current
validation result.

## End Of Run

Regenerate the run artifacts:

```bash
uv run python tools/finalize_run.py --dataset chestxray14 --compile-paper
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
