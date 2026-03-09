# Diabetic Retinopathy Grading Benchmark

## Published Results

| Method | Metric | Score | Year | Reference |
|--------|--------|-------|------|-----------|
| Gulshan et al. (Inception) | AUC | 99.1% (referable DR) | 2016 | [JAMA](https://doi.org/10.1001/jama.2016.17216) |
| APTOS 2019 1st Place | QWK | 0.936 | 2019 | Kaggle |
| APTOS 2019 Top-10 avg | QWK | 0.925 | 2019 | Kaggle |
| EyePACS baseline | QWK | 0.85 | 2015 | Kaggle |

## FDA-Cleared / Regulatory Benchmarks

| Device | Indication | Sensitivity | Specificity | Clearance |
|--------|-----------|-------------|-------------|-----------|
| IDx-DR (Digital Diagnostics) | Referable DR (>mild) | 87.2% | 90.7% | FDA De Novo (2018) |
| EyeArt (Eyenuk) | Referable DR | 95.5% | 85.0% | FDA 510(k) (2020) |
| AEYE Health | Referable DR | 95.6% | 85.7% | FDA De Novo (2022) |
| Retinalyze | DR screening | 97.0% | 91.0% | CE Mark |

*Pooled meta-analysis across 82 DR screening studies: Sensitivity 93%, Specificity 90%*

## autoresearch-medimage Results

*Run `uv run train.py --recipe aptos_winner --save-checkpoint` to add your results here.*

| Commit | QWK | Accuracy | Backbone | Recipe | Notes |
|--------|-----|----------|----------|--------|-------|
| — | — | — | — | — | *Run your first experiment!* |
