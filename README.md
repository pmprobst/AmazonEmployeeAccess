# Amazon Employee Access

A modular machine learning pipeline for the **Kaggle Amazon Employee Access** prediction challenge. The goal is to predict whether an employee’s request for access to a resource will be approved (`ACTION = 1`) or denied (`ACTION = 0`) based on role and resource attributes.

## Problem and data

- **Task:** Binary classification (approve vs. deny).
- **Data:** Categorical features (e.g. `RESOURCE`, `MGR_ID`, `ROLE_*`). Train set has an `ACTION` label; test set has an `id` for submission.
- **Metric:** ROC AUC (ranking quality).
- **Data location:** Place `train.csv` and `test.csv` in the `data/` directory. Format matches the [Kaggle competition](https://www.kaggle.com/c/amazon-employee-access-challenge) data.

## Approach

- **Preprocessing:** Shared recipes by model type: (1) **logit** — frequency encoding (in data load), target encoding, dummy encoding, normalization for penalized logistic regression; (2) **standard** — dummy encoding, normalization, optional PCA/SMOTE for SVM, KNN, Naive Bayes, MLP; (3) **tree** — dummy encoding only for random forest.
- **Models:** Penalized logistic regression (glmnet), random forest (ranger), KNN (kknn), Naive Bayes, linear SVM (kernlab), MLP (nnet). All tuned (Bayesian or grid) and evaluated with ROC AUC.
- **Output:** Kaggle-style submission files with columns `Id` and `Action` (probability of approval).

## Project structure

```
├── config.R              # Central config (seed, paths, tuning, model params)
├── run_model.R           # CLI: train one model and write submission
├── run_all_models.R      # Run multiple models and collect metrics
├── R/
│   ├── utils.R           # Logging, seeding, config loading
│   ├── data_loading.R     # Load train/test, optional frequency encoding
│   ├── preprocessing.R   # Recipe builders (logit, standard, tree)
│   ├── models.R          # Model specs and workflows
│   ├── tuning.R          # Resampling, tune_model, select_best, summarize
│   └── predict_submission.R  # fit_final_model, predict_test, write_kaggle_submission
├── data/
│   ├── train.csv
│   └── test.csv
├── output/
│   └── submission/       # Submission CSVs (default; created when run)
├── archive/              # Legacy per-model scripts (use run_model.R instead)
├── results/             # Tune RDS and metrics (when using results_dir)
└── README.md
```

## Requirements

R with the **tidymodels** ecosystem. Install from CRAN:

```r
install.packages(c(
  "tidymodels", "embed", "vroom", "tune", "themis",
  "ranger", "kknn", "kernlab", "naivebayes", "nnet", "glmnet", "discrim"
))
```

Optional: **themis** only if you set `use_smote = TRUE` in config. To capture your environment (e.g. for reproducibility), run `sessionInfo()` or use `renv::snapshot()` after `renv::init()`.

## Quickstart

1. **Install dependencies** (see Requirements above).

2. **Run a single model** from the project root:
   ```bash
   Rscript run_model.R --model penalized_logreg
   Rscript run_model.R --model rf --output_dir output/submission
   ```
   Submissions are written to `output/submission/` by default. Valid `--model` values: `penalized_logreg`, `logreg`, `rf`, `knn`, `nb`, `svm_linear`, `mlp`.

3. **Run multiple models** and write metrics to `results/all_models_metrics.csv`:
   ```bash
   Rscript run_all_models.R --output_dir results
   ```

4. **Legacy scripts:** Per-model wrappers are in `archive/` (e.g. `archive/AmazonKNN.R`). To run one from R: `source("archive/AmazonKNN.R")` from the project root. Prefer `run_model.R` or `run_all_models.R` instead.

## Configuration

Edit `config.R` to change:

- `seed`, `data_dir`, `results_dir`, `submission_dir` (default `output/submission`)
- `n_folds`, `cv_repeats`, `tune_method` (`"bayes"` or `"grid"`), `bayes_initial`, `bayes_iter`
- `use_smote`, `pca_threshold`, `rare_threshold`
- Model-specific tuning ranges (e.g. `logreg_penalty_range`, `rf_mtry_range`)

Defaults work without editing; override only what you need. If you remove `config.R`, the pipeline uses built-in defaults from `R/utils.R`.

## Results

Submission files are written to `output/submission/<model>_submission.csv` by default (and optionally to legacy names in the project root). With `run_model.R`, tune results are saved in the same output dir (e.g. `output/submission/<model>_tune_results.rds`). With `run_all_models.R`, metrics are written to the output dir as `all_models_metrics.csv`.
