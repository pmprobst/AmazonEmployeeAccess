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
├── results/              # Submission and tune outputs (created when run)
├── Amazon*.R             # Thin wrappers per model (backwards compatibility)
├── ENVIRONMENT.md        # R packages and reproducibility
└── README.md
```

## Quickstart

1. **Install dependencies** (see [ENVIRONMENT.md](ENVIRONMENT.md)):
   ```r
   install.packages(c("tidymodels", "embed", "vroom", "tune", "themis",
                     "ranger", "kknn", "kernlab", "naivebayes", "nnet", "glmnet", "discrim"))
   ```

2. **Run a single model** from the project root:
   ```bash
   Rscript run_model.R --model penalized_logreg
   Rscript run_model.R --model rf --output_dir results
   ```
   Valid `--model` values: `penalized_logreg`, `logreg`, `rf`, `knn`, `nb`, `svm_linear`, `mlp`.

3. **Run multiple models** and write metrics to `results/all_models_metrics.csv`:
   ```bash
   Rscript run_all_models.R --output_dir results
   ```

4. **Legacy scripts:** Each `Amazon*.R` script (e.g. `AmazonLogisticRegression.R`) is a thin wrapper that runs the same pipeline for that model and writes both `results/<model>_submission.csv` and the original submission filename (e.g. `LogRegModelSubmission.csv`). Run from the project root in R:
   ```r
   source("AmazonLogisticRegression.R")
   ```

## Configuration

Edit `config.R` to change:

- `seed`, `data_dir`, `results_dir`
- `n_folds`, `cv_repeats`, `tune_method` (`"bayes"` or `"grid"`), `bayes_initial`, `bayes_iter`
- `use_smote`, `pca_threshold`, `rare_threshold`
- Model-specific tuning ranges (e.g. `logreg_penalty_range`, `rf_mtry_range`)

Defaults work without editing; override only what you need.

## Results

Submission files are written to `results/<model>_submission.csv` (and optionally to the legacy names in the project root). Tune results can be saved as `results/<model>_tune_results.rds` when using `run_model.R`. Compare models via `results/all_models_metrics.csv` after running `run_all_models.R`.
