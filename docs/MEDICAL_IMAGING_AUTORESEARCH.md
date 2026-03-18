# ChestXray14 Case Study: Medical Imaging vs `karpathy/autoresearch`

This repo is intentionally close to the spirit of
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), but
medical-imaging workloads forced a few important differences.

This document is a worked example from one ChestXray14 experiment series. It is
here to explain why the repo includes optional refinement and promotion
helpers, not to claim that every dataset, backbone family, or time budget will
behave the same way.

## What We Kept

- `prepare.py` stays fixed during experiment loops.
- `train.py` remains the main experiment surface.
- the default short research loop still uses a fixed wall-clock budget
- `results.tsv` stays the untracked experiment ledger
- the point is still tight, comparable, reproducible search rather than a large
  generic orchestration framework

## What Medical Imaging Changed

### 1. The short wall-clock loop is still useful, but it is not the whole story

For language modeling, a short fixed wall-clock budget can be much closer to
"the real winner" because early optimization dynamics often dominate the
benchmark of interest. For medical imaging, especially across different
backbone families, we observed that strong candidates can mature on very
different timescales.

That means a fixed short search budget is still the right *screening* tool, but
it is not always the right *final model selection* tool.

### 2. We search on validation AUC, not bits-per-byte

The metric in this repo is task-specific:

- classification: accuracy/AUC style metrics
- multilabel classification: `mean_auc`
- segmentation: overlap metrics such as Dice or IoU

The equivalent of the `autoresearch` "fairness contract" is:

- fixed wall-clock search budget
- fixed dataset split protocol
- a task-appropriate primary metric

### 3. A promotion stage is necessary

In practice, medical-imaging search needs at least two stages:

1. a short wall-clock search to find promising architectures and recipe regions
2. a longer promotion stage to rerank top diverse candidates fairly

For stronger benchmark work, a useful pattern is actually three stages:

1. broad short-budget search
2. mid-budget refinement search around top diverse candidates
3. long-budget promotion runs for the finalists

## Concrete ChestXray14 Lesson

In a cold-start local ChestXray14 session on March 14-15, 2026, the short-loop
winner was not the long-loop winner:

- best short-budget promoted winner rerun:
  `efficientnet_b0`, `81.088263` mean AUC at 10 epochs
- promoted runner-up:
  `densenet169`, `84.726800` mean AUC at 10 epochs
- promoted runner-up:
  `resnet50`, `84.486692` mean AUC at 10 epochs
- promoted runner-up:
  alternate `efficientnet_b0` recipe, `84.802390` mean AUC at 10 epochs

The short search was still valuable because it surfaced the candidate set. But
the final ranking changed materially once we let diverse candidates train
longer.

That is the main lesson behind the optional staged workflow in this repo.

## One More Important Lesson

The staged workflow is useful partly because it can invalidate promising
intermediate winners.

In a later follow-up, several stage-2 refinement winners were promoted directly
to `20` epochs and underperformed their shorter-horizon scores. That does not
mean the earlier search was useless. It means the handoff between budgets is
itself something researchers need to validate.

In practice, this repo should be framed as a strong candidate-discovery and
staged-validation framework, not as a guarantee that one short search or one
promotion rule will always produce the final best model automatically.

## Recommended Workflow

### Stage 1: Broad Search

- keep the default `300s` contract as the baseline/autoresearch-compatible loop
- use `--time-budget 900` when you want a stronger screening budget on local or
  hosted GPUs
- optimize for breadth: backbones, loss families, augmentation presets,
  optimizer/LR/WD regions

### Stage 2: Refinement Search

- select the top diverse candidates from stage 1
- run a second search session at a longer wall-clock budget
- retune hyperparameters under that longer budget instead of freezing the
  stage-1 recipe

### Stage 3: Promotion

- run the top refined candidates for a longer epoch or wall-clock budget
- use this stage for final ranking, checkpoint selection, and any final held-out
  evaluation

## What This Means For The Repo

This repo is not trying to become a generic search framework. The goal is
still:

- minimal enough to understand
- reproducible enough to trust
- useful enough that a researcher with a labeled 2D medical-imaging dataset can
  actually run it

The main repo-level addition is therefore not "more agent." It is a staged,
budget-aware workflow that keeps the autoresearch spirit while helping
researchers discover strong candidates quickly and then validate them more
carefully when a benchmark needs it.
