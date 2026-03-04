library(limma)
library(openxlsx)
library(ggplot2)
library(dplyr)

# --- 1. General Function for DEG Analysis ---
run_limma_pipeline <- function(data_path, meta_path, column, control_grp, treatment_grp, organ) {
  
  # Load Expression Data
  expression_matrix <- read.csv(data_path, row.names = 1, check.names = FALSE)
  
  # Load Metadata
  if (grepl(".xlsx$", meta_path)) {
    meta <- openxlsx::read.xlsx(meta_path, rowNames = TRUE)
  } else {
    meta <- read.csv(meta_path, row.names = 1, check.names = FALSE)
  }
  
  if(organ == "lung") {
    rownames(expression_matrix) = rownames(meta)
  }
  
  if(organ == "liver") {
    meta["group"] <- ifelse(meta[[column]] == 1, "control", "treatment")
    column <- "group"
    control_grp <- "control"
    treatment_grp <- "treatment"
  }
  
  expression_matrix <- t(expression_matrix)
  
  colnames(expression_matrix) <- gsub("^X", "", colnames(expression_matrix))
  rownames(meta) <- gsub("^X", "", rownames(meta))
  
  # Align samples
  common_samples <- intersect(colnames(expression_matrix), rownames(meta))
  
  # SAFETY CHECK: Ensure we have enough data
  if (length(common_samples) == 0) {
    stop(paste("No matching samples found for", organ, "- check your sample IDs."))
  }
  
  expression_matrix <- expression_matrix[, common_samples]
  meta <- meta[common_samples, ]
  
  # FIX: Ensure column is treated as character then factor to avoid index issues
  meta[[column]] <- as.character(meta[[column]])
  
  # ERROR PREVENTION: Verify at least two levels exist after alignment
  unique_groups <- unique(meta[[column]])
  if (length(unique_groups) < 2) {
    stop(paste("Error in", organ, ": Only found one level (", unique_groups, 
               ") after alignment. At least two are required for DEG."))
  }
  
  # 2. Design Matrix (~0 + group)
  group <- factor(meta[[column]])
  design <- model.matrix(~0 + group)
  colnames(design) <- levels(group)
  
  # 3. Create Contrast Matrix
  # We ensure the group names used here match exactly what is in the design columns.
  contrast_formula <- paste(treatment_grp, "-", control_grp, sep = " ")
  contrasts <- makeContrasts(contrasts = contrast_formula, levels = design)
  
  # 4. Fit Linear Model and Empirical Bayes
  fit <- lmFit(expression_matrix, design)
  fit2 <- contrasts.fit(fit, contrasts)
  fit2 <- eBayes(fit2) 
  
  # 5. Extract Results with Benjamini-Hochberg (FDR) Correction
  tT <- topTable(fit2, adjust = "fdr", number = Inf)
  tT$Gene <- rownames(tT)
  
  # Save results
  output_file <- paste0("./data/degs_result_", organ, ".xlsx")
  write.xlsx(tT, file = output_file)
  cat("Successfully processed:", organ, "\n")
  
  return(tT)
}

# --- 2. Execute Experiments (Kept intact with your requested usages) ---

# A. Lung
data_lung_path <- "./data/GSE125004_lung.csv"
meta_lung_raw <- read.csv("./data/GSE125004_sample_info.csv", row.names = 1, check.names = FALSE)
cluster_col <- grep("cluster", colnames(meta_lung_raw), value = TRUE, ignore.case = TRUE)[1]

if (length(cluster_col) > 0) {
  meta_lung_raw$group <- ifelse(meta_lung_raw[[cluster_col]] != 1, "treatment", "control")
  write.csv(meta_lung_raw, "./data/temp_lung_meta.csv")
  res_lung <- run_limma_pipeline(data_lung_path, "./data/temp_lung_meta.csv", "group", "control", "treatment", "lung")
}

# B. Liver
res_liver <- run_limma_pipeline("./data/GSE145780_liver.csv", "./data/liver_meta.xlsx", "classes", 1, 0, "liver")

# C. Kidney
res_kidney <- run_limma_pipeline("./data/GSE192444_kidney.csv", "./data/kidney_meta.xlsx", "group", "control", "treatment", "kidney")

# D. Heart
res_heart <- run_limma_pipeline("./data/GSE272655_heart.csv", "./data/heart_meta.xlsx", "group", "control", "treatment", "heart")
