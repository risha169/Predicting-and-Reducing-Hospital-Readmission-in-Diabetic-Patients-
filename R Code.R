# ===============================================================
# QMH 2025 — Full Pipeline: Cleaning + EDA + Models + Hypothesis
# ===============================================================

# ----- Libraries (install once if needed) -----
# install.packages(c("readxl","ggplot2","glmnet","randomForest","xgboost","pROC"))
library(readxl)
library(ggplot2)
library(glmnet)
library(randomForest)
library(xgboost)
library(pROC)

# ===============================================================
# 1) DATA CLEANING
# ===============================================================
rawf <- read_excel("data/diabetic_data_QMH_Club_Fest_2025.xlsx")

names(rawf) <- c(
  "encounter_code","patient_code","ethnic_group","sex_identity","age_band",
  "body_weight","adm_type_code","discharge_type","adm_source_code",
  "hospital_days","insurance_code","provider_specialty",
  "lab_test_count","procedure_count","medication_count",
  "outpatient_visits","emergency_visits","inpatient_visits",
  "diagnosis_primary","diagnosis_secondary","diagnosis_tertiary",
  "diagnosis_total","glucose_test_result","A1C_result",
  paste0("med_",1:23),
  "med_change_status","diabetic_med_given","readmission_status"
)

rawf$body_weight <- NULL
rawf$insurance_code <- NULL
rawf$provider_specialty <- NULL

for (col in names(rawf)) {
  if (is.character(rawf[[col]]) || is.factor(rawf[[col]])) {
    rawf[[col]][rawf[[col]] %in% c("?","")] <- "Unknown"
    rawf[[col]][is.na(rawf[[col]])] <- "Unknown"
  } else {
    rawf[[col]][is.na(rawf[[col]])] <- median(rawf[[col]], na.rm=TRUE)
  }
}

rawf$readmitted_30 <- ifelse(rawf$readmission_status=="<30",1L,0L)


# ===============================================================
# 2) EDA  
# ===============================================================
gender_plot <- ggplot(rawf, aes(x=sex_identity, fill=factor(readmitted_30))) +
  geom_bar(position="dodge") + theme_minimal() +
  scale_fill_manual(values=c("0"="lightblue","1"="tomato"), labels=c("No","Yes")) +
  labs(title="Gender vs Readmission", x="Gender", fill="<30d")

age_plot <- ggplot(rawf, aes(x=age_band, fill=factor(readmitted_30))) +
  geom_bar(position="dodge") + theme_minimal() +
  theme(axis.text.x = element_text(angle=45,hjust=1)) +
  scale_fill_manual(values=c("0"="lightblue","1"="tomato"), labels=c("No","Yes")) +
  labs(title="Age band vs Readmission", x="Age band", fill="<30d")

readmission_plot <- ggplot(rawf, aes(x=factor(readmitted_30))) +
  geom_bar(fill=c("lightblue","tomato")) + theme_minimal() +
  scale_x_discrete(labels=c("No","Yes")) +
  labs(title="Overall Readmission Status", x="Readmitted within 30 days", y="Count")

print(gender_plot); print(age_plot); print(readmission_plot)

numeric_vars <- c("lab_test_count","medication_count","hospital_days",
                  "inpatient_visits","outpatient_visits","emergency_visits",
                  "diagnosis_total","procedure_count")
par(mfrow=c(3,3), mar=c(4,4,2,1))
for (v in numeric_vars) {
  boxplot(rawf[[v]] ~ rawf$readmitted_30, main=v, names=c("No","Yes"),
          col=c("lightblue","tomato"), ylab=v, xlab="<30d")
}
par(mfrow=c(1,1))

# quick tests
cat("\n--- T-tests for numeric variables ---\n")
for (v in numeric_vars) {
  tt <- t.test(rawf[[v]] ~ rawf$readmitted_30)
  cat(sprintf("%-18s : p = %.3g\n", v, tt$p.value))
}
cat("\n--- Chi-square for key categoricals ---\n")
for (v in c("med_change_status","A1C_result","age_band")) {
  tb <- table(rawf[[v]], rawf$readmitted_30)
  chi <- suppressWarnings(chisq.test(tb))
  cat(sprintf("%-18s : p = %.3g\n", v, chi$p.value))
}

# ===============================================================
# 3) FEATURE ENGINEERING for hypotheses (H1,H2,H3,H4,H5)
# ===============================================================
set.seed(2025)

# H1: meds >15 & LOS >4
high_meds15 <- as.integer(rawf$medication_count > 15)
long_los4   <- as.integer(rawf$hospital_days    > 4)

# H2: inpatient >1 & ED >1
high_inpt1  <- as.integer(rawf$inpatient_visits  > 1)
high_ed1    <- as.integer(rawf$emergency_visits  > 1)

# H4: Abnormal A1C & med change
a1c_abn     <- as.integer(rawf$A1C_result %in% c(">7",">8"))
medchg      <- as.integer(!(rawf$med_change_status %in% c("No","NO","Unknown")))

# H6: AMA vs Home (for hypothesis test we’ll use subset AMA/Home)
ama_flag    <- as.integer(as.character(rawf$discharge_type) %in% c("7","AMA","Against Medical Advice"))
home_flag   <- as.integer(as.character(rawf$discharge_type) %in% c("1","Home","HOME"))

# H7: High ED (>=3) × High Comorbidity (Q3)
high_ed3    <- as.integer(rawf$emergency_visits >= 3)
q3_comorb   <- as.integer(rawf$diagnosis_total >= as.numeric(quantile(rawf$diagnosis_total, 0.75, na.rm=TRUE)))

# Numeric age midpoint (kept for modeling covariate if needed)
age_lo <- suppressWarnings(as.numeric(sub(".?(\\d+).","\\1", rawf$age_band)))
age_hi <- suppressWarnings(as.numeric(sub(".-(\\d+).","\\1", rawf$age_band)))
age_mid <- ifelse(!is.na(age_lo)&!is.na(age_hi),(age_lo+age_hi)/2,age_lo)
age_mid[is.na(age_mid) & grepl("90", rawf$age_band)] <- 95
age_mid[is.na(age_mid) & grepl("80", rawf$age_band)] <- 85
age_z <- as.numeric(scale(age_mid))

# ===============================================================
# 4) MODELING DATA (used for prediction models)
# ===============================================================
M <- data.frame(
  readmitted_30    = factor(rawf$readmitted_30, levels=c(0,1)),
  
  # hypothesis pieces for prediction
  high_meds15,long_los4,high_inpt1,high_ed1,a1c_abn,medchg,ama_flag,high_ed3,q3_comorb,age_z,
  
  # strong covariates
  lab_test_count   = rawf$lab_test_count,
  medication_count = rawf$medication_count,
  hospital_days    = rawf$hospital_days,
  inpatient_visits = rawf$inpatient_visits,
  emergency_visits = rawf$emergency_visits,
  diagnosis_total  = rawf$diagnosis_total,
  procedure_count  = rawf$procedure_count,
  A1C_result       = factor(rawf$A1C_result),
  med_change_status= factor(rawf$med_change_status),
  age_band         = factor(rawf$age_band),
  adm_source_code  = factor(as.character(rawf$adm_source_code)),
  discharge_type   = factor(as.character(rawf$discharge_type))
)
M <- M[complete.cases(M), ]

# ---- Split 70 / 20 / 10 ----
set.seed(2025)
idx <- sample.int(nrow(M)); M <- M[idx, ]
n <- nrow(M)
n_train <- floor(0.7*n); n_valid <- floor(0.2*n)
train <- M[1:n_train, ]
valid <- M[(n_train+1):(n_train+n_valid), ]
test  <- M[(n_train+n_valid+1):n, ]

# ===============================================================
# 5) HELPERS
# ===============================================================
metrics_at <- function(y, p, t=0.5){
  y <- as.numeric(y); pred <- ifelse(p>=t,1L,0L)
  acc <- mean(pred==y)
  prec <- ifelse(sum(pred==1)==0,0,sum(pred==1 & y==1)/sum(pred==1))
  rec  <- sum(pred==1 & y==1)/sum(y==1)
  f1   <- ifelse(prec+rec==0, 0, 2*prec*rec/(prec+rec))
  aucv <- as.numeric(pROC::auc(pROC::roc(y, p)))
  c(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, AUC=aucv)
}
best_t <- function(y, p, ts=seq(0.2,0.6,by=0.02)){
  mm <- sapply(ts, function(t) metrics_at(y, p, t)["F1"])
  ts[which.max(mm)]
}

# ===============================================================
# 6) TRAIN 4 MODELS for PREDICTION (GLM, ENet, RF, XGB)
# ===============================================================
# formula includes interactions for H1,H2,H4; ama_flag, high_ed3, q3_comorb included as main terms
pred_form <- readmitted_30 ~ 
  high_meds15*long_los4 +              # H1 interaction
  high_inpt1*high_ed1 +                # H2 interaction
  a1c_abn*medchg +                     # H4 interaction
  ama_flag + high_ed3 + q3_comorb +    # H6 main (ama_flag), H7 components
  lab_test_count + medication_count + hospital_days +
  inpatient_visits + emergency_visits + diagnosis_total + procedure_count +
  A1C_result + med_change_status + age_band + adm_source_code + discharge_type

# matrices for glmnet/xgb
x_tr <- model.matrix(pred_form, data=train)[,-1]
x_va <- model.matrix(pred_form, data=valid)[,-1]
x_te <- model.matrix(pred_form, data=test)[,-1]
y_tr <- as.numeric(as.character(train$readmitted_30))
y_va <- as.numeric(as.character(valid$readmitted_30))
y_te <- as.numeric(as.character(test$readmitted_30))

# --- GLM (weighted)
pos_wt <- sum(train$readmitted_30==0)/sum(train$readmitted_30==1)
glm_fit <- glm(pred_form, data=train, family=binomial(),
               weights=ifelse(train$readmitted_30==1,pos_wt,1))
glm_va <- as.numeric(predict(glm_fit, newdata=valid, type="response"))
t_glm <- best_t(y_va, glm_va)
glm_te <- as.numeric(predict(glm_fit, newdata=test, type="response"))
glm_metrics <- metrics_at(y_te, glm_te, t_glm)

# --- Elastic Net (alpha=0.5)
set.seed(31)
cv_en <- glmnet::cv.glmnet(x_tr, y_tr, family="binomial", alpha=0.5,
                           weights=ifelse(y_tr==1,pos_wt,1), nfolds=5)
en_fit <- glmnet::glmnet(x_tr, y_tr, family="binomial", alpha=0.5,
                         lambda=cv_en$lambda.min, weights=ifelse(y_tr==1,pos_wt,1))
en_va <- as.numeric(predict(en_fit, newx=x_va, type="response"))
t_en <- best_t(y_va, en_va)
en_te <- as.numeric(predict(en_fit, newx=x_te, type="response"))
en_metrics <- metrics_at(y_te, en_te, t_en)

# --- Random Forest (prediction only)
set.seed(41)
rf_fit <- randomForest::randomForest(pred_form, data=train, ntree=600, importance=TRUE)
rf_va <- as.numeric(predict(rf_fit, newdata=valid, type="prob")[,2])
t_rf <- best_t(y_va, rf_va)
rf_te <- as.numeric(predict(rf_fit, newdata=test, type="prob")[,2])
rf_metrics <- metrics_at(y_te, rf_te, t_rf)

# --- XGBoost (prediction only)
set.seed(51)
dtr <- xgboost::xgb.DMatrix(x_tr, label=y_tr)
xgb_params <- list(objective="binary:logistic", eval_metric="aucpr",
                   eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                   scale_pos_weight=sum(y_tr==0)/sum(y_tr==1))
xgb_fit <- xgboost::xgb.train(params=xgb_params, data=dtr, nrounds=700, verbose=0)
xgb_va <- as.numeric(predict(xgb_fit, xgboost::xgb.DMatrix(x_va)))
t_xgb <- best_t(y_va, xgb_va)
xgb_te <- as.numeric(predict(xgb_fit, xgboost::xgb.DMatrix(x_te)))
xgb_metrics <- metrics_at(y_te, xgb_te, t_xgb)

# ---- Performance table
perf <- rbind(
  GLM = round(glm_metrics,3),
  ENet= round(en_metrics,3),
  RF  = round(rf_metrics,3),
  XGB = round(xgb_metrics,3)
)
cat("\n==============================\nModel Performance (TEST)\n==============================\n")
print(perf)
write.csv(perf, "Model_Performance_70_20_10.csv", row.names=FALSE)

# ===============================================================
# 7) HYPOTHESIS TESTS — ONLY GLM & ENet (OR + 95% CI + p)
#     H1,H2,H3 from the main GLM using interaction terms
#     H5 (AMA vs Home): subset GLM; H6 (ED>=3 × Q3) own GLM
# ===============================================================

## ---- Helper to extract OR, CI, p from a GLM term
or_ci <- function(fit, term){
  sm <- summary(fit); cf <- sm$coefficients
  if(!(term %in% rownames(cf))) return(c(OR=NA,L=NA,U=NA,p=NA))
  est <- cf[term,"Estimate"]; se <- sqrt(vcov(fit)[term,term])
  c(OR=exp(est), L=exp(est-1.96*se), U=exp(est+1.96*se), p=cf[term,"Pr(>|z|)"])
}

## ---- H1, H2, H3 from main GLM
H1_glm <- or_ci(glm_fit, "high_meds15:long_los4")
H2_glm <- or_ci(glm_fit, "high_inpt1:high_ed1")
H4_glm <- or_ci(glm_fit, "a1c_abn:medchg")

## ---- H4: AMA vs Home (subset to those two types for clean contrast)
sub_h6 <- subset(M, ama_flag==1 | as.character(discharge_type) %in% c("1","Home","HOME"))
pos_wt_h6 <- sum(sub_h6$readmitted_30==0)/sum(sub_h6$readmitted_30==1)
glm_h6 <- glm(readmitted_30 ~ ama_flag +
                lab_test_count + medication_count + hospital_days +
                inpatient_visits + emergency_visits + diagnosis_total +
                procedure_count + A1C_result + med_change_status + age_band +
                adm_source_code,
              data=sub_h6, family=binomial(),
              weights=ifelse(sub_h6$readmitted_30==1,pos_wt_h6,1))
H6_glm <- or_ci(glm_h6, "ama_flag")

## ---- H5: High ED (>=3) × High Comorbidity (Q3) interaction with mains
glm_h7 <- glm(readmitted_30 ~ high_ed3*q3_comorb +
                lab_test_count + medication_count + hospital_days +
                inpatient_visits + emergency_visits + diagnosis_total +
                procedure_count + A1C_result + med_change_status + age_band +
                adm_source_code + discharge_type,
              data=M, family=binomial(),
              weights=ifelse(M$readmitted_30==1, sum(M$readmitted_30==0)/sum(M$readmitted_30==1), 1))
H7_glm <- or_ci(glm_h7, "high_ed3:q3_comorb")

## ---- ENet support: positive coefficient for the same terms
b_full <- as.matrix(coef(en_fit))[,1]
has_pos <- function(name) ifelse(name %in% names(b_full), b_full[name] > 1e-6, FALSE)
EN_H1 <- has_pos("high_meds15:long_los4")
EN_H2 <- has_pos("high_inpt1:high_ed1")
EN_H4 <- has_pos("a1c_abn:medchg")

# ENet for H4 (subset design)
x_h6 <- model.matrix(readmitted_30 ~ ama_flag +
                       lab_test_count + medication_count + hospital_days +
                       inpatient_visits + emergency_visits + diagnosis_total +
                       procedure_count + A1C_result + med_change_status + age_band +
                       adm_source_code,
                     data=sub_h6)[,-1]
y_h6 <- as.numeric(as.character(sub_h6$readmitted_30))
set.seed(321)
cv_h6 <- cv.glmnet(x_h6, y_h6, family="binomial", alpha=0.5,
                   weights=ifelse(y_h6==1, sum(y_h6==0)/sum(y_h6==1), 1), nfolds=5)
en_h6 <- glmnet(x_h6, y_h6, family="binomial", alpha=0.5, lambda=cv_h6$lambda.min,
                weights=ifelse(y_h6==1, sum(y_h6==0)/sum(y_h6==1), 1))
b_h6 <- as.matrix(coef(en_h6))[,1]
EN_H6 <- ("ama_flag" %in% names(b_h6)) && (b_h6["ama_flag"] > 1e-6)

# ENet for H6 (own design with interaction)
x_h7 <- model.matrix(readmitted_30 ~ high_ed3*q3_comorb +
                       lab_test_count + medication_count + hospital_days +
                       inpatient_visits + emergency_visits + diagnosis_total +
                       procedure_count + A1C_result + med_change_status + age_band +
                       adm_source_code + discharge_type,
                     data=M)[,-1]
y_h7 <- as.numeric(as.character(M$readmitted_30))
set.seed(322)
cv_h7 <- cv.glmnet(x_h7, y_h7, family="binomial", alpha=0.5,
                   weights=ifelse(y_h7==1, sum(y_h7==0)/sum(y_h7==1), 1), nfolds=5)
en_h7 <- glmnet(x_h7, y_h7, family="binomial", alpha=0.5, lambda=cv_h7$lambda.min,
                weights=ifelse(y_h7==1, sum(y_h7==0)/sum(y_h7==1), 1))
b_h7 <- as.matrix(coef(en_h7))[,1]
EN_H7 <- ("high_ed3:q3_comorb" %in% names(b_h7)) && (b_h7["high_ed3:q3_comorb"] > 1e-6)

## ---- Assemble hypothesis table (require direction OR>1 and p<0.05 for GLM)
## ---- Assemble hypothesis table (using p-value; show "Negatively associated" for OR<1; remove CI)
mk_row <- function(code, title, g_or, en_ok){
  if (is.na(g_or["p"])) {
    glms <- FALSE
    assoc_text <- "NA"
  } else if (as.numeric(g_or["p"]) < 0.05) {
    if (as.numeric(g_or["OR"]) > 1) {
      assoc_text <- "✓ Supported"
      glms <- TRUE
    } else if (as.numeric(g_or["OR"]) < 1) {
      assoc_text <- "🔻 Negatively associated"
      glms <- FALSE
    } else {
      assoc_text <- "✗ Not supported"
      glms <- FALSE
    }
  } else {
    assoc_text <- "✗ Not supported"
    glms <- FALSE
  }
  
  data.frame(
    Hypothesis   = paste0(code, ": ", title),
    GLM_OR       = round(as.numeric(g_or["OR"]), 2),
    GLM_P        = ifelse(is.na(g_or["p"]), NA, signif(as.numeric(g_or["p"]), 3)),
    GLM_Result   = assoc_text,
    ENet_Support = ifelse(en_ok, "✓ Supported", "✗ Not supported"),
    Overall      = ifelse(glms & en_ok, "✅ SUPPORTED",
                          ifelse(assoc_text == "🔻 Negatively associated", "🔻 Negatively associated", "❌ NOT SUPPORTED")),
    check.names  = FALSE
  )
}

hyp_tab <- rbind(
  mk_row("H1","High meds (>15) × Long stay (>4d) interaction", H1_glm, EN_H1),
  mk_row("H2","High inpatient (>1) × High ED (>1) interaction", H2_glm, EN_H2),
  mk_row("H4","Abnormal A1C × Med change interaction",          H4_glm, EN_H4),
  mk_row("H6","AMA discharge vs Home (subset)",                 H6_glm, EN_H6),
  mk_row("H7","High ED (≥3) × High Comorbidity (Q3) interaction", H7_glm, EN_H7)
)

cat("\n==============================\nHypothesis Significance (GLM + ENet)\n==============================\n")
print(hyp_tab, row.names=FALSE)
write.csv(hyp_tab, "Hypothesis_Significance_H1_H2_H4_H6_H7.csv", row.names=FALSE)

cat("\nFiles saved:\n - Model_Performance_70_20_10.csv\n - Hypothesis_Significance_H1_H2_H4_H6_H7.csv\n")

8) FEATURE IMPORTANCE PLOTS
> # ===============================================================
> cat("\n==============================\nGenerating Feature Importance Plots...\n==============================\n")

# 8) FEATURE IMPORTANCE PLOTS
# ===============================================================
cat("\n==============================\nGenerating Feature Importance Plots...\n==============================\n")

# --- 1. GLM Importance (by abs(z-value)) ---
try({
  imp_glm_df <- data.frame(summary(glm_fit)$coefficients)
  # Correctly identify the 'z value' column, which is typically the 3rd column
  names(imp_glm_df)[3] <- "z_value" 
  imp_glm_df$Importance <- abs(imp_glm_df$z_value)
  imp_glm_df$Feature <- rownames(imp_glm_df)
  imp_glm_df <- imp_glm_df[imp_glm_df$Feature != "(Intercept)", ]
  imp_glm_df <- imp_glm_df[order(-imp_glm_df$Importance), ]
  imp_glm_top20 <- head(imp_glm_df, 20)
  
  plot_glm_imp <- ggplot(imp_glm_top20, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat="identity", fill="steelblue") +
    coord_flip() +
    labs(title = "GLM Feature Importance (Top 20 by |z-value|)", x = "Feature", y = "Absolute z-value") +
    theme_minimal()
  print(plot_glm_imp)
}, silent=TRUE)


# --- 2. Elastic Net Importance (by abs(coefficient)) ---
try({
  imp_en_raw <- as.matrix(coef(en_fit, s = "lambda.min"))
  imp_en_df <- data.frame(
    Feature = rownames(imp_en_raw),
    Importance = abs(imp_en_raw[, 1])
  )
  imp_en_df <- imp_en_df[imp_en_df$Feature != "(Intercept)" & imp_en_df$Importance > 1e-6, ]
  imp_en_df <- imp_en_df[order(-imp_en_df$Importance), ]
  n_en <- min(20, nrow(imp_en_df))
  imp_en_top <- head(imp_en_df, n_en)
  
  plot_en_imp <- ggplot(imp_en_top, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat="identity", fill="darkorange") +
    coord_flip() +
    labs(title = paste("Elastic Net Feature Importance (Top", n_en, "Non-Zero Coeffs)"), x = "Feature", y = "Absolute Coefficient") +
    theme_minimal()
  print(plot_en_imp)
}, silent=TRUE)


# --- 3. Random Forest Importance (by MeanDecreaseGini) ---
try({
  imp_rf_raw <- randomForest::importance(rf_fit)
  imp_rf_df <- data.frame(
    Feature = rownames(imp_rf_raw),
    Importance = imp_rf_raw[, "MeanDecreaseGini"]
  )
  imp_rf_df <- imp_rf_df[order(-imp_rf_df$Importance), ]
  imp_rf_top20 <- head(imp_rf_df, 20)
  
  plot_rf_imp <- ggplot(imp_rf_top20, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat="identity", fill="forestgreen") +
    coord_flip() +
    labs(title = "Random Forest Importance (Top 20 by MeanDecreaseGini)", x = "Feature", y = "Mean Decrease Gini") +
    theme_minimal()
  print(plot_rf_imp)
}, silent=TRUE)


# --- 4. XGBoost Importance (by Gain) ---
try({
  imp_xgb_raw <- xgboost::xgb.importance(feature_names = colnames(x_tr), model = xgb_fit)
  imp_xgb_top20 <- head(imp_xgb_raw, 20) # Already sorted
  
  plot_xgb_imp <- ggplot(imp_xgb_top20, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_bar(stat="identity", fill="firebrick") +
    coord_flip() +
    labs(title = "XGBoost Importance (Top 20 by Gain)", x = "Feature", y = "Importance (Gain)") +
    theme_minimal()
  print(plot_xgb_imp)
}, silent=TRUE)

cat("\nFeature importance plots generated.\n")


cat("\nFiles saved:\n - Model_Performance_70_20_10.csv\n - Hypothesis_Significance_H1_H2_H4_H6_H7.csv\n")
