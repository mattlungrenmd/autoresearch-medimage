# autoresearch-vision

This is an experiment to have an LLM autonomously optimize image classification and segmentation models.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloaders, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `./data/` contains the dataset. If not, tell the human to run `uv run prepare.py` (or `uv run prepare.py --dataset voc` for segmentation).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

### Task Modes

- **Classification** (default): `uv run train.py` — trains on CIFAR-10, measures `val_accuracy`
- **Classification (CIFAR-100)**: `uv run train.py --dataset cifar100` — harder classification
- **Segmentation**: `uv run train.py --task segmentation --dataset voc` — trains on Pascal VOC, measures `val_miou`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, augmentation strategy, etc.
- Switch between classification and segmentation tasks.
- Try different architectures: ResNets, ConvNeXt-style blocks, Vision Transformers, MobileNet-style depthwise separables, attention mechanisms, squeeze-and-excitation, etc.
- Try different optimizers: SGD, Adam, AdamW, etc.
- Try data augmentation: CutOut, MixUp, CutMix, RandAugment, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (torch, torchvision, numpy, Pillow, matplotlib, pandas).
- Modify the evaluation functions. `evaluate_classification` and `evaluate_segmentation` in `prepare.py` are the ground truth metrics.

### Goals

- **Classification**: Get the **highest val_accuracy** (%)
- **Segmentation**: Get the **highest val_miou** (%)

Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. A 0.1% accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1% improvement from deleting code? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this (classification):

```
---
val_accuracy:     93.25%
val_loss:         0.234567
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4560.2
num_steps:        11720
num_params_M:     2.3
num_epochs:       15
dataset:          cifar10
task:             classification
```

Or for segmentation:

```
---
val_miou:         45.20%
val_pixel_acc:    88.50%
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     8230.5
num_steps:        2340
num_params_M:     17.3
num_epochs:       8
dataset:          voc
task:             segmentation
```

You can extract the key metric from the log file:

```
grep "^val_accuracy:\|^val_miou:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	metric	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy or val_miou achieved — use 0.00 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	metric	memory_gb	status	description
a1b2c3d	91.50	4.5	keep	baseline cifar10
b2c3d4e	92.30	4.6	keep	increase LR to 0.2 with cosine schedule
c3d4e5f	90.10	4.5	discard	switch to GeLU activation
d4e5f6g	0.00	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_accuracy:\|^val_miou:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the metric improved (higher accuracy or mIoU), you "advance" the branch, keeping the git commit
9. If the metric is equal or worse, you git reset back to where you started

### Ideas to explore (classification)

- **Architecture**: Deeper models, wider models, squeeze-and-excitation blocks, depthwise separable convolutions, attention mechanisms, ConvNeXt blocks, MobileNet-style inverted bottlenecks
- **Optimizers**: AdamW with cosine LR, one-cycle LR, LARS, label smoothing
- **Augmentation**: CutOut, CutMix, MixUp, RandAugment, AutoAugment, random erasing
- **Training tricks**: Stochastic depth, drop path, EMA, progressive resizing
- **Efficiency**: Knowledge distillation (self-distillation), channel pruning

### Ideas to explore (segmentation)

- **Architecture**: DeepLab-style ASPP, FPN, PSPNet pooling, attention gates in U-Net, dilated convolutions, lightweight decoders
- **Loss functions**: Dice loss, focal loss, boundary loss, combined CE + Dice
- **Augmentation**: Color jitter, random scale, random rotation, multi-scale training
- **Training tricks**: Auxiliary losses at intermediate scales, deep supervision

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The loop runs until the human interrupts you, period. If you run out of ideas, think harder.
