# Bring Your Own Data

`autoresearch-medimage` can train on local datasets without editing source
code when the dataset follows the repo contract.

## Folder Layout

```text
data/
  your_dataset/
    dataset.yaml
    splits.csv
    images/
      ...
```

## `dataset.yaml`

Required fields:

```yaml
name: "your_dataset"
task: "classification"        # classification | multilabel | segmentation
metric: "auc"                 # classification: accuracy | auc | f1 | macro_f1 | quadratic_kappa
num_classes: 2
class_names: ["negative", "positive"]
in_channels: 1                # 1 or 3
image_size: 224
mean: [0.485]                 # optional
std: [0.229]                  # optional
description: "Short dataset description"   # optional
```

Notes:

- classification also supports `metric: f1` and `metric: macro_f1`
- `multilabel` also supports `metric: macro_f1` and `metric: micro_f1`
- segmentation support in public v1 is binary only (`num_classes: 2`)
- `mean` and `std` default to ImageNet-style values if omitted

## `splits.csv`

Classification:

```csv
split,image_path,label
train,images/example_1.pgm,0
val,images/example_2.pgm,1
```

Multilabel:

```csv
split,image_path,labels
train,images/example_1.png,1|0|0|1
val,images/example_2.png,0|1|0|0
```

Segmentation:

```csv
split,image_path,mask_path
train,images/example_1.png,masks/example_1.png
val,images/example_2.png,masks/example_2.png
```

Notes:

- relative paths are resolved from the dataset directory
- `labels` must contain one pipe-separated `0` or `1` value per class
- `split` must include at least one `train` row and one `val` row

## Validation

Validate the contract before training:

```bash
uv run prepare.py --dataset your_dataset
```

This does not download anything for custom datasets. It checks the manifest,
CSV schema, and referenced files and then prints a summary.

## Training

Once validation passes:

```bash
uv run train.py --dataset your_dataset
```

You can also copy one of the recipe templates in `recipes/` and point it at
your dataset for a reproducible baseline.
