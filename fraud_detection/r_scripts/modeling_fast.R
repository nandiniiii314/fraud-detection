# R Modeling Script (Fast)
# Trains Logistic Regression + Random Forest in R
# Run: Rscript r_scripts/modeling_fast.R creditcard.csv

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) > 0) args[1] else "creditcard.csv"

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)][1])
base_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
r_lib <- file.path(base_dir, "r_library")
dir.create(r_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(r_lib, .libPaths()))

env_int <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(value) || value <= 0) default else value
}

stratified_sample <- function(data, max_rows) {
  if (nrow(data) <= max_rows) return(data)

  split_df <- split(data, data$Class)
  sizes <- vapply(split_df, nrow, integer(1))
  props <- sizes / sum(sizes)
  keep <- pmax(1L, floor(max_rows * props))

  sampled <- mapply(function(chunk, n_keep) {
    if (nrow(chunk) <= n_keep) return(chunk)
    chunk[sample.int(nrow(chunk), n_keep), , drop = FALSE]
  }, split_df, keep, SIMPLIFY = FALSE)

  out <- do.call(rbind, sampled)
  out[sample.int(nrow(out)), , drop = FALSE]
}

majority_ratio <- env_int("R_MODEL_LEGIT_RATIO", 15L)
balanced_rows <- env_int("R_MODEL_BALANCED_ROWS", 12000L)
rf_trees <- env_int("R_RF_TREES", 50L)
test_rows <- env_int("R_MODEL_TEST_ROWS", 30000L)

pkgs <- c("randomForest", "caret", "pROC", "ggplot2", "dplyr", "ROSE")
for (p in pkgs) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

cat("=== R Modeling: Credit Card Fraud Detection (Fast) ===\n")
cat("Reading:", csv_path, "\n\n")

df <- read.csv(csv_path)
df$Class <- as.factor(df$Class)

cat("Preprocessing...\n")
df$Amount <- scale(df$Amount)
df$Time <- scale(df$Time)

set.seed(42)
train_idx <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_df <- df[train_idx, ]
test_df <- df[-train_idx, ]

if (nrow(test_df) > test_rows) {
  test_df <- stratified_sample(test_df, test_rows)
}

fraud_train <- train_df[train_df$Class == 1, ]
legit_train <- train_df[train_df$Class == 0, ]
legit_keep <- min(nrow(legit_train), nrow(fraud_train) * majority_ratio)

train_core <- rbind(
  fraud_train,
  legit_train[sample.int(nrow(legit_train), legit_keep), , drop = FALSE]
)
train_core <- train_core[sample.int(nrow(train_core)), , drop = FALSE]

cat("Fast mode settings:\n")
cat("Legit-to-fraud ratio:", majority_ratio, ":1\n")
cat("Balanced train rows :", balanced_rows, "\n")
cat("Random Forest trees:", rf_trees, "\n")
cat("Test rows cap       :", test_rows, "\n\n")

cat("Balancing classes with ROSE...\n")
rose_rows <- max(balanced_rows, 2L * nrow(fraud_train))
train_bal <- ovun.sample(
  Class ~ ., data = train_core, method = "both",
  N = rose_rows, p = 0.5, seed = 42
)$data

cat("Core train size    :", nrow(train_core), "\n")
cat("Balanced train size:", nrow(train_bal), "\n")
cat("  Fraud:", sum(train_bal$Class == 1), "| Legit:", sum(train_bal$Class == 0), "\n")
cat("Test size          :", nrow(test_df), "\n\n")

out_dir <- "r_output"
dir.create(out_dir, showWarnings = FALSE)

eval_model <- function(model, test_data, model_name) {
  pred_class <- predict(model, newdata = test_data)
  pred_prob <- tryCatch(
    predict(model, newdata = test_data, type = "prob")[, 2],
    error = function(e) as.numeric(pred_class) - 1
  )

  cm <- confusionMatrix(pred_class, test_data$Class, positive = "1")
  roc_curve <- roc(test_data$Class, pred_prob, quiet = TRUE)

  cat("-------------------------------------\n")
  cat("Model    :", model_name, "\n")
  cat("Accuracy :", round(cm$overall["Accuracy"] * 100, 2), "%\n")
  cat("Precision:", round(cm$byClass["Precision"] * 100, 2), "%\n")
  cat("Recall   :", round(cm$byClass["Recall"] * 100, 2), "%\n")
  cat("F1-Score :", round(cm$byClass["F1"] * 100, 2), "%\n")
  cat("ROC-AUC  :", round(auc(roc_curve) * 100, 2), "%\n")
  cat("Confusion Matrix:\n")
  print(cm$table)
  cat("\n")

  list(cm = cm, roc = roc_curve, pred_prob = pred_prob, name = model_name)
}

cat("Training Logistic Regression (R)...\n")
lr_model <- glm(Class ~ ., data = train_bal, family = binomial())
lr_prob <- predict(lr_model, newdata = test_df, type = "response")
lr_pred <- factor(ifelse(lr_prob > 0.5, 1, 0), levels = levels(test_df$Class))
lr_cm <- confusionMatrix(lr_pred, test_df$Class, positive = "1")
lr_roc <- roc(test_df$Class, lr_prob, quiet = TRUE)

cat("-------------------------------------\n")
cat("Model    : Logistic Regression (R)\n")
cat("Accuracy :", round(lr_cm$overall["Accuracy"] * 100, 2), "%\n")
cat("Precision:", round(lr_cm$byClass["Precision"] * 100, 2), "%\n")
cat("Recall   :", round(lr_cm$byClass["Recall"] * 100, 2), "%\n")
cat("F1-Score :", round(lr_cm$byClass["F1"] * 100, 2), "%\n")
cat("ROC-AUC  :", round(auc(lr_roc) * 100, 2), "%\n")
cat("Confusion Matrix:\n")
print(lr_cm$table)
cat("\n")

cat("Training Random Forest (R)...\n")
set.seed(42)
rf_model <- randomForest(Class ~ ., data = train_bal, ntree = rf_trees, importance = TRUE)
rf_results <- eval_model(rf_model, test_df, "Random Forest (R)")

cat("Generating ROC curve plot...\n")
roc_lr <- data.frame(FPR = 1 - lr_roc$specificities, TPR = lr_roc$sensitivities, Model = "Logistic Regression")
roc_rf <- data.frame(FPR = 1 - rf_results$roc$specificities, TPR = rf_results$roc$sensitivities, Model = "Random Forest")
roc_all <- rbind(roc_lr, roc_rf)

p_roc <- ggplot(roc_all, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#484f58") +
  scale_color_manual(values = c("Logistic Regression" = "#1f6feb", "Random Forest" = "#3fb950")) +
  annotate("text", x = 0.7, y = 0.15,
           label = paste0("LR AUC: ", round(auc(lr_roc) * 100, 2), "%\nRF AUC: ", round(auc(rf_results$roc) * 100, 2), "%"),
           color = "#e6edf3", size = 4) +
  labs(title = "ROC Curves - R Models", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 13) +
  theme(
    plot.background = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid = element_line(color = "#30363d"),
    text = element_text(color = "#e6edf3"),
    axis.text = element_text(color = "#8b949e"),
    legend.background = element_rect(fill = "#161b22"),
    plot.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_roc_curves.png"), p_roc, width = 7, height = 5, dpi = 120)

cat("Generating feature importance plot...\n")
imp_df <- as.data.frame(importance(rf_model))
imp_df$Feature <- rownames(imp_df)
imp_df <- imp_df %>% arrange(desc(MeanDecreaseGini)) %>% head(15)

p_imp <- ggplot(imp_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "#1f6feb", alpha = 0.85) +
  coord_flip() +
  labs(title = "Random Forest Feature Importance (R)", x = NULL, y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid = element_line(color = "#30363d"),
    text = element_text(color = "#e6edf3"),
    axis.text = element_text(color = "#8b949e"),
    plot.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_feature_importance.png"), p_imp, width = 7, height = 5, dpi = 120)

cat("Generating metrics comparison plot...\n")
metrics_df <- data.frame(
  Model = rep(c("Logistic Regression", "Random Forest"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1"), 2),
  Value = c(
    round(lr_cm$overall["Accuracy"] * 100, 2),
    round(lr_cm$byClass["Precision"] * 100, 2),
    round(lr_cm$byClass["Recall"] * 100, 2),
    round(lr_cm$byClass["F1"] * 100, 2),
    round(rf_results$cm$overall["Accuracy"] * 100, 2),
    round(rf_results$cm$byClass["Precision"] * 100, 2),
    round(rf_results$cm$byClass["Recall"] * 100, 2),
    round(rf_results$cm$byClass["F1"] * 100, 2)
  )
)

p_metrics <- ggplot(metrics_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.6) +
  geom_text(aes(label = paste0(Value, "%")), position = position_dodge(0.6),
            vjust = -0.3, size = 3.2, color = "#e6edf3") +
  scale_fill_manual(values = c("Logistic Regression" = "#1f6feb", "Random Forest" = "#3fb950")) +
  ylim(0, 115) +
  labs(title = "Model Metrics Comparison (R)", x = NULL, y = "Score (%)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.background = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid = element_line(color = "#30363d"),
    text = element_text(color = "#e6edf3"),
    axis.text = element_text(color = "#8b949e"),
    legend.background = element_rect(fill = "#161b22"),
    plot.title = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_metrics_comparison.png"), p_metrics, width = 7, height = 4, dpi = 120)

cat("=== R Modeling Complete! ===\n")
cat("Plots saved to:", out_dir, "\n")
