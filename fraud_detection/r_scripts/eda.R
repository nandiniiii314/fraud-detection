# EDA Script: Fraud Detection
# Run: Rscript r_scripts/eda.R creditcard.csv

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
  keep  <- pmax(1L, floor(max_rows * props))

  sampled <- mapply(function(chunk, n_keep) {
    if (nrow(chunk) <= n_keep) return(chunk)
    chunk[sample.int(nrow(chunk), n_keep), , drop = FALSE]
  }, split_df, keep, SIMPLIFY = FALSE)

  out <- do.call(rbind, sampled)
  out[sample.int(nrow(out)), , drop = FALSE]
}

plot_rows <- env_int("R_EDA_PLOT_ROWS", 50000L)
box_rows  <- env_int("R_EDA_BOX_ROWS", 30000L)
corr_rows <- env_int("R_EDA_CORR_ROWS", 3000L)

pkgs <- c("ggplot2", "dplyr", "scales", "reshape2")
for (p in pkgs) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

cat("=== R EDA: Credit Card Fraud Detection ===\n")
cat("Reading dataset from:", csv_path, "\n")

df <- read.csv(csv_path)
df$Class <- as.factor(df$Class)

cat("\n--- Dataset Overview ---\n")
cat("Rows        :", nrow(df), "\n")
cat("Columns     :", ncol(df), "\n")
cat("Fraud cases :", sum(df$Class == 1), "\n")
cat("Legit cases :", sum(df$Class == 0), "\n")
cat("Fraud rate  :", round(mean(df$Class == 1) * 100, 3), "%\n")
cat("Missing vals:", sum(is.na(df)), "\n")

cat("\n--- Amount Summary ---\n")
print(summary(df$Amount))

cat("\n--- Class Summary ---\n")
print(table(df$Class))

cat("\nFast mode settings:\n")
cat("Plot sample rows :", plot_rows, "\n")
cat("Boxplot rows     :", box_rows, "\n")
cat("Correlation rows :", corr_rows, "\n")

out_dir <- "r_output"
dir.create(out_dir, showWarnings = FALSE)

set.seed(42)
plot_df <- stratified_sample(df, plot_rows)

cat("\nGenerating Plot 1: Class Distribution...\n")
class_df <- df %>%
  group_by(Class) %>%
  summarise(Count = n(), .groups = "drop") %>%
  mutate(Label = ifelse(Class == 1, "Fraud", "Legitimate"),
         Pct   = round(Count / sum(Count) * 100, 3))

p1 <- ggplot(class_df, aes(x = Label, y = Count, fill = Label)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste0(Count, "\n(", Pct, "%)")),
            vjust = -0.3, size = 4, color = "#e6edf3") +
  scale_fill_manual(values = c("Fraud" = "#f85149", "Legitimate" = "#238636")) +
  scale_y_continuous(labels = comma) +
  labs(title = "Class Distribution", subtitle = "Legitimate vs Fraudulent Transactions",
       x = NULL, y = "Count") +
  theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid       = element_line(color = "#30363d"),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e"),
    legend.position  = "none",
    plot.title       = element_text(face = "bold", size = 15),
    plot.subtitle    = element_text(color = "#8b949e")
  )
ggsave(file.path(out_dir, "r_class_distribution.png"), p1, width = 6, height = 4, dpi = 120)

cat("Generating Plot 2: Amount Distribution...\n")
df_amt <- plot_df[plot_df$Amount < 2000, ]

p2 <- ggplot(df_amt, aes(x = Amount, fill = Class)) +
  geom_histogram(bins = 60, alpha = 0.75, position = "identity", color = NA) +
  scale_fill_manual(values = c("0" = "#238636", "1" = "#f85149"),
                    labels = c("0" = "Legitimate", "1" = "Fraud")) +
  scale_x_continuous(labels = dollar) +
  labs(title = "Transaction Amount Distribution (< $2000)",
       x = "Amount ($)", y = "Count", fill = NULL) +
  theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid       = element_line(color = "#30363d"),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e"),
    legend.background= element_rect(fill = "#161b22"),
    plot.title       = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_amount_distribution.png"), p2, width = 8, height = 4, dpi = 120)

cat("Generating Plot 3: Time Distribution...\n")
plot_df$Hour <- plot_df$Time / 3600

p3 <- ggplot(plot_df, aes(x = Hour, fill = Class)) +
  geom_histogram(bins = 48, alpha = 0.75, position = "identity", color = NA) +
  scale_fill_manual(values = c("0" = "#1f6feb", "1" = "#f85149"),
                    labels = c("0" = "Legitimate", "1" = "Fraud")) +
  labs(title = "Transaction Frequency by Hour", x = "Hour", y = "Count", fill = NULL) +
  theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid       = element_line(color = "#30363d"),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e"),
    legend.background= element_rect(fill = "#161b22"),
    plot.title       = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_time_distribution.png"), p3, width = 8, height = 4, dpi = 120)

cat("Generating Plot 4: Boxplot...\n")
df_box <- df[df$Amount < quantile(df$Amount, 0.99), ]
df_box <- stratified_sample(df_box, box_rows)

p4 <- ggplot(df_box, aes(x = Class, y = Amount, fill = Class)) +
  geom_boxplot(outlier.color = "#e3b341", outlier.size = 0.5, alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#238636", "1" = "#f85149")) +
  scale_x_discrete(labels = c("0" = "Legitimate", "1" = "Fraud")) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Transaction Amount by Class (99th percentile)",
       x = NULL, y = "Amount ($)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    panel.grid       = element_line(color = "#30363d"),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e"),
    legend.position  = "none",
    plot.title       = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_amount_boxplot.png"), p4, width = 6, height = 4, dpi = 120)

cat("Generating Plot 5: Correlation heatmap...\n")
sample_df <- stratified_sample(df, corr_rows)
sample_df$Class <- as.numeric(as.character(sample_df$Class))
key_cols <- c("Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "Class")
corr_mat <- cor(sample_df[, key_cols])
corr_melt <- reshape2::melt(corr_mat)

p5 <- ggplot(corr_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "#30363d") +
  geom_text(aes(label = round(value, 2)), size = 2.8, color = "#e6edf3") +
  scale_fill_gradient2(low = "#f85149", mid = "#161b22", high = "#1f6feb",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap (Key Features)", x = NULL, y = NULL, fill = "r") +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#161b22", color = NA),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e", size = 9),
    axis.text.x      = element_text(angle = 45, hjust = 1),
    legend.background= element_rect(fill = "#161b22"),
    plot.title       = element_text(face = "bold")
  )
ggsave(file.path(out_dir, "r_correlation.png"), p5, width = 7, height = 6, dpi = 120)

cat("\n=== EDA Complete! ===\n")
cat("Plots saved to:", out_dir, "\n")
cat("Files:\n")
for (f in list.files(out_dir, pattern = "\\.png$")) cat(" -", f, "\n")
