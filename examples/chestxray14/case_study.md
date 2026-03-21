# ChestXray14 Case Study

This note summarizes what changed when we adapted
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) to a
medical-imaging benchmark.

It is one worked example, not a claim that every dataset will behave the same
way.

## What Stayed The Same

- `prepare.py` stays fixed during search runs
- `train.py` is still the file used for each scored run
- short runs are compared under a fixed wall-clock budget
- `results.tsv` remains the untracked results log

## What Changed For Medical Imaging

- the metric is task-specific (`mean_auc`, F1, Dice, QWK, accuracy), not bits
  per byte
- the data contract has to handle labeled image folders, split files, and
  segmentation masks
- the search space is different: backbones, image sizes, augmentation, loss,
  optimizer, and regularization choices matter much more than tokenization or
  sequence length
- slower image models can rerank later, so the repo includes optional helpers
  for adaptive search, refinement, and longer follow-up runs

## What Happened On ChestXray14

The short search was useful for finding promising candidates quickly, but the
ranking changed once we trained several finalists longer.

In one local experiment series on March 14-15, 2026:

- short-run promoted winner rerun:
  `efficientnet_b0`, `81.088263` mean AUC at 10 epochs
- promoted runner-up:
  `densenet169`, `84.726800` mean AUC at 10 epochs
- promoted runner-up:
  `resnet50`, `84.486692` mean AUC at 10 epochs
- promoted runner-up:
  alternate `efficientnet_b0` recipe, `84.802390` mean AUC at 10 epochs

The short run still did its job: it surfaced the candidate set. It just was not
enough on its own to rank the final winner.

## Practical Workflow

For medical-imaging benchmarks, the most useful pattern so far is:

1. broad short-budget search
2. refinement search around the top diverse candidates
3. longer follow-up runs for the finalists

That is why the repo ships:

- `tools/run_search_session.py`
- `tools/build_refinement_space.py`
- `tools/run_full_train.py`
- `tools/run_full_train_queue.py`

## Takeaway

For this kind of workload, the repo is strongest as:

- a candidate-discovery tool
- a way to compare strong alternatives under a consistent short budget
- a staged validation workflow for follow-up runs

It should not be read as a promise that one short search or one promotion rule
will always pick the final best model automatically.
