# GPU Setup Guide - Free & Paid Options

## Free GPU Options

### Google Colab (Free Tier)
- **GPU**: T4 (16GB VRAM) - free, limited hours
- **Notebook options**:
  - [Quick start](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/notebooks/colab_quickstart.ipynb) for smoke tests and BYOD
  - [Single benchmark demo](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/examples/chestxray14/colab_demo.ipynb) for one ChestXray14 run
  - [Autoresearch session](https://colab.research.google.com/github/mattlungrenmd/autoresearch-medimage/blob/main/notebooks/colab_autoresearch.ipynb) for a named multi-run search
- **Setup**:
```python
# In a Colab notebook (public repo, no GitHub token needed):
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]

!rm -rf /content/autoresearch-medimage /content/autoresearch-medimage-main /content/repo.zip
!wget -q https://github.com/mattlungrenmd/autoresearch-medimage/archive/refs/heads/main.zip -O /content/repo.zip
!unzip -q /content/repo.zip -d /content
!mv /content/autoresearch-medimage-main /content/autoresearch-medimage
%cd /content/autoresearch-medimage
!uv sync

# Fastest trial run:
!uv run prepare.py --dataset toy_binary
!uv run train.py --recipe toy_binary_smoke --time-budget 5
```
- **No private setup**: no GitHub token, SSH key, or local install required once the repo is public
- **Local upload fallback**: if your Colab Google account and GitHub account differ, upload the notebook and repo zip locally and run the same flow without GitHub auth inside the runtime
- **Bring your own data**: upload a zip to Colab or mount Drive, then run `uv run prepare.py --data /content/my_images`
- **Best ChestX-ray14 search pattern**: mount Drive, stage the dataset to local `/content` once, then run a named `--session` with frontier-based 300-second search waves and periodic Drive backups
- **Limits**: ~4 hrs/day GPU, sessions disconnect after idle

### Kaggle Notebooks (Free Tier)
- **GPU**: T4 x2 (16GB each) or P100 - 30 hrs/week
- **Setup**:
```python
# Kaggle notebook - datasets auto-available
!git clone https://github.com/mattlungrenmd/autoresearch-medimage
%cd autoresearch-medimage
!pip install -e .
# Kaggle datasets are pre-mounted at /kaggle/input/
!python train.py --recipe chestxray14_baseline --save-checkpoint
```
- **Advantage**: Kaggle datasets load instantly (no download needed)

### Lightning AI Studios (Free Tier)
- **GPU**: T4 or A10G - free credits for new users
- **Setup**: Create a studio, clone repo, run normally
- **Advantage**: Persistent environment, VS Code in browser

### Google Cloud Free Credits ($300 for 90 days)
- **GPU**: T4, A100 available
- **Setup**: Create a VM with GPU, SSH in, run normally

## Paid GPU Options (Recommended for Serious Work)

### Lambda Labs
- **A100 80GB**: ~$1.10/hr
- **H100**: ~$2.49/hr
- Best for overnight autoresearch runs

### RunPod
- **A100 80GB**: ~$1.19/hr
- **H100**: ~$3.29/hr
- Serverless and pod options

### Vast.ai
- **Community GPUs**: from $0.15/hr
- **A100**: ~$0.80/hr
- Cheapest option, variable availability

### AWS/GCP/Azure
- Spot instances for cheaper rates
- Best for production workloads

## Recommended GPU by Task

| Task | Min VRAM | Recommended | Batch Size |
|------|----------|-------------|------------|
| CIFAR / Small datasets | 4GB | T4 (16GB) | 128 |
| Chest X-ray classification | 8GB | T4 (16GB) | 16-32 |
| ChestX-ray14 (full) | 16GB | A100 (40GB) | 32-64 |
| Brain MRI segmentation | 8GB | T4 (16GB) | 8-16 |
| DR grading (384px) | 12GB | A10G (24GB) | 16 |
| DR grading (512px) | 16GB | A100 (40GB) | 8-16 |
| ViT/Large models | 24GB+ | A100 (80GB) | 8-16 |

## Running Overnight (Agent Session)

For the full autoresearch experience, keep a Claude/Codex session attached to
this repo and point it at `program.md`.

This repository does **not** ship a standalone overnight scheduler that edits
`train.py` on its own. The autonomy comes from the agent session following the
repo contract:

- `prepare.py` stays fixed
- `train.py` is the experiment surface
- each run uses the fixed 300-second budget
- the agent keeps or discards ideas based on the metric and `results.tsv`

A practical pattern is:

```bash
# Start a persistent shell session
tmux new -s autoresearch

# Open Claude or Codex in this repo and ask it to follow program.md
# while redirecting experiment output to run.log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t autoresearch
```

Rough overnight capacity for 8 hours:

- **T4**: about 90 fixed-budget experiments
- **A100**: about 95 fixed-budget experiments

Those numbers assume an agent is continuously driving the loop, not a passive
shell script.
