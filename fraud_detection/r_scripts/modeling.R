# ── R Modeling Script ────────────────────────────────────
# Trains Logistic Regression + Random Forest in R
# Run: Rscript r_scripts/modeling.R creditcard.csv

args     <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) > 0) args[1] else "creditcard.csv"

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)][1])
base_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
r_lib <- file.path(base_dir, "r_library")
dir.create(r_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(r_lib, .libPaths()))

pkgs <- c("randomForest", "caret", "pROC", "ggplot2", "dplyr", "ROSE")
for (p in pkgs) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

cat("=== R Modeling: Credit Card Fraud Detection ===\n")
cat("Reading:", csv_path, "\n\n")

df       <- read.csv(csv_path)
df$Class <- as.factor(df$Class)

# ── Preprocess ───────────────────────────────────────────
cat("Preprocessing...\n")
df$Amount <- scale(df$Amount)
df$Time   <- scale(df$Time)

set.seed(42)
train_idx <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_df  <- df[ train_idx, ]
test_df   <- df[-train_idx, ]

# Balance with ROSE (oversampling)
cat("Balancing classes with ROSE...\n")
train_bal <- ovun.sample(Class ~ ., data = train_df,
                          method = "over", N = nrow(train_df)*2, seed = 42)$data
cat("Balanced train size:", nrow(train_bal), "\n")
cat("  Fraud:", sum(train_bal$Class==1), "| Legit:", sum(train_bal$Class==0), "\n\n")

out_dir <- "r_output"
dir.create(out_dir, showWarnings = FALSE)

eval_model <- function(model, test_data, model_name) {
  pred_class <- predict(model, newdata = test_data)
  pred_prob  <- tryCatch(
    predict(model, newdata = test_data, type = "prob")[,2],
    error = function(e) as.numeric(pred_class) - 1
  )

  cm  <- confusionMatrix(pred_class, test_data$Class, positive = "1")
  roc <- roc(test_data$Class, pred_prob, quiet = TRUE)

  cat("─────────────────────────────────────\n")
  cat("Model    :", model_name, "\n")
  cat("Accuracy :", round(cm$overall["Accuracy"]*100, 2), "%\n")
  cat("Precision:", round(cm$byClass["Precision"]*100, 2), "%\n")
  cat("Recall   :", round(cm$byClass["Recall"]*100, 2), "%\n")
  cat("F1-Score :", round(cm$byClass["F1"]*100, 2), "%\n")
  cat("ROC-AUC  :", round(auc(roc)*100, 2), "%\n")
  cat("Confusion Matrix:\n")
  print(cm$table)
  cat("\n")

  list(cm=cm, roc=roc, pred_prob=pred_prob, name=model_name)
}

# ── Logistic Regression ──────────────────────────────────
cat("Training Logistic Regression (R)...\n")
lr_model <- glm(Class ~ ., data = train_bal, family = binomial())
lr_prob  <- predict(lr_model, newdata = test_df, type = "response")
lr_pred  <- as.factor(ifelse(lr_prob > 0.5, 1, 0))
lr_cm    <- confusionMatrix(lr_pred, test_df$Class, positive = "1")
lr_roc   <- roc(test_df$Class, lr_prob, quiet = TRUE)

cat("─────────────────────────────────────\n")
cat("Model    : Logistic Regression (R)\n")
cat("Accuracy :", round(lr_cm$overall["Accuracy"]*100, 2), "%\n")
cat("Precision:", round(lr_cm$byClass["Precision"]*100, 2), "%\n")
cat("Recall   :", round(lr_cm$byClass["Recall"]*100, 2), "%\n")
cat("F1-Score :", round(lr_cm$byClass["F1"]*100, 2), "%\n")
cat("ROC-AUC  :", round(auc(lr_roc)*100, 2), "%\n")
cat("Confusion Matrix:\n")
print(lr_cm$table)
cat("\n")

# ── Random Forest ────────────────────────────────────────
cat("Training Random Forest (R) — this may take 1-2 minutes...\n")
rf_model <- randomForest(Class ~ ., data = train_bal,
                          ntree = 100, importance = TRUE, seed = 42)
rf_results <- eval_model(rf_model, test_df, "Random Forest (R)")

# ── Plot 1: ROC Curves ───────────────────────────────────
cat("Generating ROC curve plot...\n")
roc_lr <- data.frame(FPR=1-lr_roc$specificities, TPR=lr_roc$sensitivities, Model="Logistic Regression")
roc_rf <- data.frame(FPR=1-rf_results$roc$specificities, TPR=rf_results$roc$sensitivities, Model="Random Forest")
roc_all <- rbind(roc_lr, roc_rf)

p_roc <- ggplot(roc_all, aes(x=FPR, y=TPR, color=Model)) +
  geom_line(linewidth=1.2) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="#484f58") +
  scale_color_manual(values=c("Logistic Regression"="#1f6feb","Random Forest"="#3fb950")) +
  annotate("text", x=0.7, y=0.15,
           label=paste0("LR AUC: ", round(auc(lr_roc)*100,2),"%\nRF AUC: ", round(auc(rf_results$roc)*100,2),"%"),
           color="#e6edf3", size=4) +
  labs(title="ROC Curves — R Models", x="False Positive Rate", y="True Positive Rate") +
  theme_minimal(base_size=13) +
  theme(
    plot.background  = element_rect(fill="#0d1117", color=NA),
    panel.background = element_rect(fill="#161b22", color=NA),
    panel.grid       = element_line(color="#30363d"),
    text             = element_text(color="#e6edf3"),
    axis.text        = element_text(color="#8b949e"),
    legend.background= element_rect(fill="#161b22"),
    plot.title       = element_text(face="bold")
  )
ggsave(file.path(out_dir, "r_roc_curves.png"), p_roc, width=7, height=5, dpi=120)

# ── Plot 2: RF Feature Importance ───────────────────────
cat("Generating feature importance plot...\n")
imp_df <- as.data.frame(importance(rf_model))
imp_df$Feature <- rownames(imp_df)
imp_df <- imp_df %>% arrange(desc(MeanDecreaseGini)) %>% head(15)

p_imp <- ggplot(imp_df, aes(x=reorder(Feature, MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat="identity", fill="#1f6feb", alpha=0.85) +
  coord_flip() +
  labs(title="Random Forest Feature Importance (R)", x=NULL, y="Mean Decrease Gini") +
  theme_minimal(base_size=12) +
  theme(
    plot.background  = element_rect(fill="#0d1117", color=NA),
    panel.background = element_rect(fill="#161b22", color=NA),
    panel.grid       = element_line(color="#30363d"),
    text             = element_text(color="#e6edf3"),
    axis.text        = element_text(color="#8b949e"),
    plot.title       = element_text(face="bold")
  )
ggsave(file.path(out_dir, "r_feature_importance.png"), p_imp, width=7, height=5, dpi=120)

# ── Plot 3: Metrics Comparison Bar ──────────────────────
cat("Generating metrics comparison plot...\n")
metrics_df <- data.frame(
  Model  = rep(c("Logistic Regression","Random Forest"), each=4),
  Metric = rep(c("Accuracy","Precision","Recall","F1"), 2),
  Value  = c(
    round(lr_cm$overall["Accuracy"]*100,2),
    round(lr_cm$byClass["Precision"]*100,2),
    round(lr_cm$byClass["Recall"]*100,2),
    round(lr_cm$byClass["F1"]*100,2),
    round(rf_results$cm$overall["Accuracy"]*100,2),
    round(rf_results$cm$byClass["Precision"]*100,2),
    round(rf_results$cm$byClass["Recall"]*100,2),
    round(rf_results$cm$byClass["F1"]*100,2)
  )
)

p_metrics <- ggplot(metrics_df, aes(x=Metric, y=Value, fill=Model)) +
  geom_bar(stat="identity", position="dodge", width=0.6) +
  geom_text(aes(label=paste0(Value,"%")), position=position_dodge(0.6),
            vjust=-0.3, size=3.2, color="#e6edf3") +
  scale_fill_manual(values=c("Logistic Regression"="#1f6feb","Random Forest"="#3fb950")) +
  ylim(0, 115) +
  labs(title="Model Metrics Comparison (R)", x=NULL, y="Score (%)") +
  theme_minimal(base_size=13) +
  theme(
    plot.background  = element_rect(fill="#0d1117", color=NA),
    panel.background = element_rect(fill="#161b22", color=NA),
    panel.grid       = element_line(color="#30363d"),
    text             = element_text(color="#e6edf3"),
    axis.text        = element_text(color="#8b949e"),
    legend.background= element_rect(fill="#161b22"),
    plot.title       = element_text(face="bold")
  )
ggsave(file.path(out_dir, "r_metrics_comparison.png"), p_metrics, width=7, height=4, dpi=120)

cat("=== R Modeling Complete! ===\n")
cat("Plots saved to:", out_dir, "\n")
