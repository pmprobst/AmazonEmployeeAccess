# Environment and reproducibility

This project uses **R** with the **tidymodels** ecosystem. All random seeds are controlled via `config.R` (default seed: 348) so that runs are reproducible.

## Required R packages

Install from CRAN:

- **tidymodels** (workflows, parsnip, recipes, rsample, tune, yardstick, dials)
- **embed** (target encoding: `step_lencode_mixed`)
- **vroom** (fast CSV I/O)
- **tune** (Bayesian tuning: `tune_bayes`)
- **themis** (optional; for SMOTE if `use_smote = TRUE` in config)
- **ranger** (random forest engine)
- **kknn** (KNN engine)
- **kernlab** (SVM engine)
- **naivebayes** (Naive Bayes engine; parsnip uses `naive_Bayes` from discrim + naivebayes)
- **nnet** (MLP engine)
- **glmnet** (penalized logistic regression)
- **discrim** (Naive Bayes parsnip spec)

Install with:

```r
install.packages(c(
  "tidymodels", "embed", "vroom", "tune", "themis",
  "ranger", "kknn", "kernlab", "naivebayes", "nnet", "glmnet", "discrim"
))
```

## Capturing your environment

To record your R version and package versions for reproducibility, run in R:

```r
sessionInfo()
# Or, to write to a file:
sink("sessionInfo.txt")
sessionInfo()
sink()
```

You can also use **renv** to create a lockfile:

```r
install.packages("renv")
renv::init()
# After installing packages and running the pipeline:
renv::snapshot()
```

Then others can run `renv::restore()` to install the same versions.
