# Archive

Legacy or unused files moved here to keep the project root clean.

- **Amazon*.R** — Per-model wrapper scripts. The same workflows are now run via:
  - `Rscript run_model.R --model <name>`
  - `Rscript run_all_models.R`
  You can still run a script from here by sourcing it from the project root (e.g. `source("archive/AmazonKNN.R")`) if you need the old entry point.
