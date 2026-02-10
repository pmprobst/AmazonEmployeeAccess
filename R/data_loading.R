# =============================================================================
# data_loading.R — Load train/test data and basic checks
# =============================================================================

#' Load training and test data from the data directory.
#'
#' Reads train and test CSVs, sets ACTION to factor for classification,
#' and optionally adds frequency-encoded features. Performs basic checks
#' (dimensions, presence of ACTION/id).
#'
#' @param config List with at least \code{data_dir}, \code{train_file}, \code{test_file}.
#' @param add_freq_encoding If TRUE, add frequency encoding features (for logreg/linear models).
#' @return List with \code{train} and \code{test} tibbles.
#' @export
load_data <- function(config, add_freq_encoding = FALSE) {
  data_dir <- config$data_dir %||% "data"
  train_path <- file.path(data_dir, config$train_file %||% "train.csv")
  test_path <- file.path(data_dir, config$test_file %||% "test.csv")

  if (!file.exists(train_path)) stop("Training file not found: ", train_path)
  if (!file.exists(test_path)) stop("Test file not found: ", test_path)

  train_data <- vroom::vroom(train_path)
  train_data$ACTION <- factor(train_data$ACTION)
  test_data <- vroom::vroom(test_path)

  log_msg("Training data: ", nrow(train_data), " rows, ", ncol(train_data), " columns.")
  log_msg("Test data: ", nrow(test_data), " rows, ", ncol(test_data), " columns.")
  log_msg("Class distribution: ", paste(names(table(train_data$ACTION)), "=", table(train_data$ACTION), collapse = ", "))

  if (add_freq_encoding) {
    out <- add_frequency_encoding(train_data, test_data)
    train_data <- out$train
    test_data <- out$test
    log_msg("Added frequency encoding. Train columns: ", ncol(train_data), ", test columns: ", ncol(test_data))
  }

  list(train = train_data, test = test_data)
}

#' Add frequency encoding for each categorical predictor.
#' Uses training-set frequencies; test set gets 0 for unseen levels.
add_frequency_encoding <- function(train_data, test_data) {
  predictor_cols <- setdiff(names(train_data), c("id", "ACTION"))

  for (col in predictor_cols) {
    freq_table <- table(train_data[[col]])
    original_type <- class(train_data[[col]])[1]

    level_values <- if (original_type %in% c("numeric", "integer", "double")) {
      as.numeric(names(freq_table))
    } else {
      names(freq_table)
    }

    freq_df <- data.frame(
      level = level_values,
      freq = as.numeric(freq_table),
      stringsAsFactors = FALSE
    )
    freq_col_name <- paste0(col, "_freq")
    names(freq_df)[2] <- freq_col_name

    if (original_type %in% c("numeric", "integer", "double")) {
      train_data[[col]] <- as.numeric(train_data[[col]])
      test_data[[col]] <- as.numeric(test_data[[col]])
    }

    train_data <- dplyr::left_join(train_data, freq_df, by = setNames("level", col))
    test_data <- dplyr::left_join(test_data, freq_df, by = setNames("level", col))
    test_data[[freq_col_name]][is.na(test_data[[freq_col_name]])] <- 0
  }

  list(train = train_data, test = test_data)
}
