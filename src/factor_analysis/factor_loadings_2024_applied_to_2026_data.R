library(lavaan)
library(dplyr)
library(semTools)



load_and_rescale_data2024 <- function(data_split_path) {
  data <- merge(
    read.csv(paste("/Users/jheitz/git/luha-prolific-study/data/processed_combined/", data_split_path, sep="")),
    read.csv("/Users/jheitz/git/luha-prolific-study/data/processed_combined/data/language_task_scores.csv"),
    by = "study_submission_id"
  )
  print("Dimension of data")
  print(dim(data))
  
  # Invert specified columns (higher = better)
  cols_to_invert <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                      "avg_reaction_speed", "place_the_beads_total_extra_moves", 
                      "fill_the_grid_total_time", 
                      "typeskill_time", "clickskill_time", "dragskill_time")
  data[cols_to_invert] <- data[cols_to_invert] * -1
  
  # Rescale specified columns (to make variances more similar)
  cols_to_rescale <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                       "fill_the_grid_total_time", "typeskill_time", 
                       "clickskill_time", "dragskill_time", "avg_reaction_speed")
  cols_to_rescale_factor <- c(1000, 1000, 1000, 1000, 1000, 1000, 10)
  
  for (i in seq_along(cols_to_rescale)) {
    data[[cols_to_rescale[i]]] <- data[[cols_to_rescale[i]]] / cols_to_rescale_factor[i]
  }
  
  study_submission_ids = data$study_submission_id
  
  # Select the specified columns
  cols_to_keep <- c('connect_the_dots_I_time_msec',
                    'connect_the_dots_II_time_msec', 'wordlist_correct_words'
                    ,'avg_reaction_speed'
                    , 'place_the_beads_total_extra_moves'
                    ,'box_tapping_total_correct', 'fill_the_grid_total_time'
                    ,'wordlist_delayed_correct_words', 'wordlist_recognition_correct_words'
                    ,'digit_sequence_1_correct_series', 'digit_sequence_2_correct_series'
                    , 'clickskill_time', 'dragskill_time'
                    ,"semantic_fluency_score","phonemic_fluency_score",  "picture_naming_score"
                    , "study_submission_id"
  )
  data <- data[cols_to_keep]
  
  # Keep only complete rows
  data_complete_rows <- data[complete.cases(data), ]
  
  study_submission_ids_complete_rows = data_complete_rows$study_submission_id
  
  data <- subset(data, select = -c(study_submission_id))
  data_complete_rows <- subset(data_complete_rows, select = -c(study_submission_id))
  
  return(list(data = data, data_complete_rows = data_complete_rows, study_submission_ids = study_submission_ids, study_submission_ids_complete_rows = study_submission_ids_complete_rows))
}


data_2024_list <- load_and_rescale_data2024("data/acs_outcomes_imputed.csv")
data_2024 <- data_2024_list$data
data_2024_complete <- data_2024_list$data_complete_rows
data_2024_complete_ids <- data_2024_list$study_submission_ids_complete_rows

load_and_rescale_data <- function(data_split_path) {
  data <- merge(
    read.csv(paste("/Users/jheitz/git/luha-prolific-study/data2026/processed_combined/", data_split_path, sep="")),
    read.csv("/Users/jheitz/git/luha-prolific-study/data2026/processed_combined/data/language_task_scores.csv"),
    by = "study_submission_id"
  )
  print("Dimension of data")
  print(dim(data))
  
  # Invert specified columns (higher = better)
  cols_to_invert <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                      "avg_reaction_speed", "place_the_beads_total_extra_moves", 
                      "fill_the_grid_total_time", 
                      "typeskill_time", "clickskill_time", "dragskill_time")
  data[cols_to_invert] <- data[cols_to_invert] * -1
  
  # Rescale specified columns (to make variances more similar)
  cols_to_rescale <- c("connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec", 
                       "fill_the_grid_total_time", "typeskill_time", 
                       "clickskill_time", "dragskill_time", "avg_reaction_speed")
  cols_to_rescale_factor <- c(1000, 1000, 1000, 1000, 1000, 1000, 10)
  
  for (i in seq_along(cols_to_rescale)) {
    data[[cols_to_rescale[i]]] <- data[[cols_to_rescale[i]]] / cols_to_rescale_factor[i]
  }
  
  study_submission_ids = data$study_submission_id
  
  # rename language tasks to be consistent with 2024 data
  names(data)[names(data) == 'phonemic_fluency_f_score'] <- 'phonemic_fluency_score'
  
  # Select the specified columns
  cols_to_keep <- c('connect_the_dots_I_time_msec',
                    'connect_the_dots_II_time_msec', 'wordlist_correct_words'
                    ,'avg_reaction_speed'
                    , 'place_the_beads_total_extra_moves'
                    ,'box_tapping_total_correct', 'fill_the_grid_total_time'
                    ,'wordlist_delayed_correct_words', 'wordlist_recognition_correct_words'
                    ,'digit_sequence_1_correct_series', 'digit_sequence_2_correct_series'
                    , 'clickskill_time', 'dragskill_time'
                    ,"semantic_fluency_score","phonemic_fluency_score",  "picture_naming_score"
                    , "study_submission_id"
  )
  print(colnames(data))
  print(cols_to_keep)
  data <- data[cols_to_keep]
  
  # Keep only complete rows
  data_complete_rows <- data[complete.cases(data), ]
  
  study_submission_ids_complete_rows = data_complete_rows$study_submission_id
  
  data <- subset(data, select = -c(study_submission_id))
  data_complete_rows <- subset(data_complete_rows, select = -c(study_submission_id))
  
  return(list(data = data, data_complete_rows = data_complete_rows, study_submission_ids = study_submission_ids, study_submission_ids_complete_rows = study_submission_ids_complete_rows))
}

# ------------------------------------------------------------
# 1. Load Wave 1 model and data
# ------------------------------------------------------------
fit_original <- readRDS("~/git/luha-prolific-study/src/resources/model_fitted_2026-02-13-1011.rds")

modelSyntax <- '
  memory =~ wordlist_correct_words + wordlist_delayed_correct_words + wordlist_recognition_correct_words
  language =~ semantic_fluency_score + phonemic_fluency_score + picture_naming_score
  speed =~ avg_reaction_speed + fill_the_grid_total_time + clickskill_time + dragskill_time
  executive_function =~ connect_the_dots_I_time_msec + connect_the_dots_II_time_msec + digit_sequence_1_correct_series + digit_sequence_2_correct_series
  digit_sequence_1_correct_series ~~ digit_sequence_2_correct_series
  speed =~ connect_the_dots_I_time_msec
'

# ------------------------------------------------------------
# 2. Load 2026 data
# ------------------------------------------------------------
data_2026_list <- load_and_rescale_data("data/acs_outcomes_imputed.csv")
data_2026_complete <- data_2026_list$data_complete_rows
ids_2026_complete <- data_2026_list$study_submission_ids_complete_rows

# ------------------------------------------------------------
# 3. Identify wave2-returning, wave2-new, and wave3 participants
#    using the waves_and_ids.csv mapping file
# ------------------------------------------------------------
waves_and_ids <- read.csv(
  "/Users/jheitz/git/luha-prolific-study/data2026/processed_combined/data/waves_and_ids.csv",
  stringsAsFactors = FALSE
)

# Make sure the ID column types match for joining/lookup
waves_and_ids$study_submission_id <- as.character(waves_and_ids$study_submission_id)
ids_2026_complete_chr <- as.character(ids_2026_complete)

# longitudinal_ids that appeared in wave1 — used to flag returning wave2 participants (cohort A)
wave1_longitudinal_ids <- waves_and_ids$longitudinal_id[
  waves_and_ids$wave == "wave1" & !is.na(waves_and_ids$longitudinal_id) &
    waves_and_ids$longitudinal_id != ""
]
# longitudinal ids in wave3 -> cohort B
wave3_longitudinal_ids <- waves_and_ids$longitudinal_id[
  waves_and_ids$wave == "wave3" & !is.na(waves_and_ids$longitudinal_id) &
    waves_and_ids$longitudinal_id != ""
]
wave1_longitudinal_ids <- unique(wave1_longitudinal_ids)
wave3_longitudinal_ids <- unique(wave3_longitudinal_ids)

# Build a lookup for each submission id -> wave and longitudinal_id
lookup <- waves_and_ids[, c("study_submission_id", "longitudinal_id", "wave")]
lookup <- lookup[!duplicated(lookup$study_submission_id), ]

idx <- match(ids_2026_complete_chr, lookup$study_submission_id)
participant_wave        <- lookup$wave[idx]
participant_longitudinal <- lookup$longitudinal_id[idx]

# Flags for each subgroup
is_wave2 <- !is.na(participant_wave) & participant_wave == "wave2"
is_wave3 <- !is.na(participant_wave) & participant_wave == "wave3"

# cohort A = wave2 returning
is_wave2_returning <- is_wave2 &
  !is.na(participant_longitudinal) &
  participant_longitudinal != "" &
  participant_longitudinal %in% wave1_longitudinal_ids

# cohort B
is_cohortB <- !is.na(participant_longitudinal) &
  participant_longitudinal != "" &
  participant_longitudinal %in% wave3_longitudinal_ids
is_cohortB_baseline <- is_cohortB & is_wave2


is_wave2_new <- is_wave2 & !is_wave2_returning

# Subset data
data_wave2_returning <- data_2026_complete[is_wave2_returning, ]
data_wave2_new       <- data_2026_complete[is_wave2_new, ]
data_wave3           <- data_2026_complete[is_wave3, ]
data_cohortB         <- data_2026_complete[is_cohortB, ]
data_cohortB_baseline <- data_2026_complete[is_cohortB_baseline, ]

ids_wave2_returning <- ids_2026_complete[is_wave2_returning]
ids_wave2_new       <- ids_2026_complete[is_wave2_new]
ids_wave3           <- ids_2026_complete[is_wave3]
ids_cohortB         <- ids_2026_complete[is_cohortB]
ids_cohortB_basline <- ids_2026_complete[is_cohortB_baseline]

cat("Wave2 returning:", nrow(data_wave2_returning),
    " | Wave2 new:",   nrow(data_wave2_new),
    " | Cohort B:",   nrow(data_cohortB),
    " | Wave3:",       nrow(data_wave3), "\n")

# Sanity check: any complete-row participants not classified?
n_unclassified <- sum(!(is_wave2_returning | is_wave2_new | is_wave3))
if (n_unclassified > 0) {
  cat("WARNING:", n_unclassified,
      "complete-row participants were not classified into any subgroup.\n")
}

# ------------------------------------------------------------
# 4. Refit model structure on each Wave 2 / Wave 3 subsample
#    MATCH the settings of the original fit: std.lv = TRUE, MLR
#    Do NOT pass start = coef(fit_original) — let lavaan fit cleanly.
#    This tests whether the same structure holds; loadings are
#    re-estimated and can be compared to Wave 1.
# ------------------------------------------------------------
fit_wave2_returning <- cfa(
  modelSyntax,
  data      = data_wave2_returning,
  std.lv    = TRUE,
  estimator = "MLR"
)

fit_wave2_new <- cfa(
  modelSyntax,
  data      = data_wave2_new,
  std.lv    = TRUE,
  estimator = "MLR"
)

fit_wave3 <- cfa(
  modelSyntax,
  data      = data_wave3,
  std.lv    = TRUE,
  estimator = "MLR"
)

fit_cohortB <- cfa(
  modelSyntax,
  data      = data_cohortB,
  std.lv    = TRUE,
  estimator = "MLR"
)

cat("\n--- Wave 2 returning participants ---\n")
print(fitMeasures(fit_wave2_returning,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))

cat("\n--- Wave 2 new participants ---\n")
print(fitMeasures(fit_wave2_new,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))

cat("\n--- Wave 3 participants ---\n")
print(fitMeasures(fit_wave3,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))

cat("\n--- Cohort B (wave 2 and 3) participants ---\n")
print(fitMeasures(fit_cohortB,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))

# ------------------------------------------------------------
# 5. Also report fit with loadings
#    FIXED to Wave 1 values — this is the stricter test of
#    "same measurement model".
# ------------------------------------------------------------
pe <- parameterEstimates(fit_original, standardized = FALSE) %>%
  filter(op == "=~")

fixed_model_lines <- pe %>%
  mutate(line = sprintf("  %s =~ %.6f*%s", lhs, est, rhs)) %>%
  pull(line)

# Add the residual covariance from the original model
modelSyntax_fixed <- paste(
  c(fixed_model_lines,
    "  digit_sequence_1_correct_series ~~ digit_sequence_2_correct_series"),
  collapse = "\n"
)

fit_wave2_returning_fixed <- cfa(
  modelSyntax_fixed,
  data      = data_wave2_returning,
  std.lv    = FALSE,   # with fixed loadings, don't also fix lv variance
  estimator = "MLR"
)
fit_wave2_new_fixed <- cfa(
  modelSyntax_fixed,
  data      = data_wave2_new,
  std.lv    = FALSE,
  estimator = "MLR"
)
fit_wave3_fixed <- cfa(
  modelSyntax_fixed,
  data      = data_wave3,
  std.lv    = FALSE,
  estimator = "MLR"
)
fit_cohortB_fixed <- cfa(
  modelSyntax_fixed,
  data      = data_cohortB,
  std.lv    = FALSE,
  estimator = "MLR"
)

cat("\n--- Wave 2 returning, loadings fixed to Wave 1 ---\n")
print(fitMeasures(fit_wave2_returning_fixed,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))
cat("\n--- Wave 2 new, loadings fixed to Wave 1 ---\n")
print(fitMeasures(fit_wave2_new_fixed,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))
cat("\n--- Wave 3, loadings fixed to Wave 1 ---\n")
print(fitMeasures(fit_wave3_fixed,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))
cat("\n--- Cohort B, loadings fixed to Wave 1 ---\n")
print(fitMeasures(fit_cohortB_fixed,
                  c("cfi.robust", "tli.robust", "rmsea.robust", "srmr")))


# ------------------------------------------------------------
# 6. Generate factor scores using the ORIGINAL Wave 1 model
#    (ensures Wave 1 and Wave 2/3 scores are on the same metric)
# ------------------------------------------------------------
factorScores <- data.frame(
  lavPredict(fit_original, newdata = data_2026_complete)
)
factorScores$study_submission_id <- ids_2026_complete



# check: does lavPredict also use the original intercepts?
lavInspect(fit_original, "meanstructure")  # false

# are the scores centered to the new mean? --> no, they are between 0.11 and 0.23
mean(factorScores$language)
mean(factorScores$memory)
mean(factorScores$speed)
mean(factorScores$executive_function)

# what if I apply the original fit to the 2024 data  -> all means are 0, SD around 0.9 each
# also, the predicted scores are really exactly like the ones I originally calculated in 2024
factorScores2024 <- data.frame(lavPredict(fit_original, newdata = data_2024_complete))
factorScores2024$study_submission_id <- data_2024_complete_ids
mean(factorScores2024$language)
mean(factorScores2024$memory)
mean(factorScores2024$speed)
mean(factorScores2024$executive_function)
sd(factorScores2024$language)
sd(factorScores2024$memory)
sd(factorScores2024$speed)
sd(factorScores2024$executive_function)

# what if I predict only for one observation. does this give the same result as 
# predicting on all 2026 cases (which would be further evidence there is no new centering)?
# --> gives the same results, so that's good
data.frame(lavPredict(fit_original, newdata = data_2024_complete[1:2,]))
data.frame(lavPredict(fit_original, newdata = data_2024_complete))[1:2,]

# where does predict get the mean from to center by?
# fit_original@SampleStats@mean --> is exactly the same as mean(data_2024_complete$wordlist_correct_words) but different to mean(data_2026_complete$wordlist_correct_words)
fit_original@SampleStats@mean
mean(data_2024_complete$wordlist_correct_words)
mean(data_2026_complete$wordlist_correct_words)



# proper test of invariance (instead of the fitting above)

# Test for Cohort A followup
cat("\n--- Cohort A followup invariance tests ---\n")
df_base <- transform(data_2024_complete, grp = "baseline")
df_fu   <- transform(data_wave2_returning, grp = "followup")
combined <- rbind(df_base, df_fu)

fit_config <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE)
fit_metric <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = "loadings")
fit_scalar <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = c("loadings", "intercepts"))
compared_fit <- compareFit(fit_config, fit_metric, fit_scalar)
summary(compared_fit)


# Test for Cohort B (baseline and followup --> not strictly correct because each observation is assumed to be independent, which it is not)
cat("\n--- Cohort B baseline and followup invariance tests vs. 2024 data ---\n")
df_base <- transform(data_2024_complete, grp = "baseline")
df_fu   <- transform(data_cohortB, grp = "cohortB")
combined <- rbind(df_base, df_fu)

fit_config <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE)
fit_metric <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = "loadings")
fit_scalar <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = c("loadings", "intercepts"))
compared_fit <- compareFit(fit_config, fit_metric, fit_scalar)
summary(compared_fit)

# Test for Cohort B baseline only
cat("\n--- Cohort B baseline only invariance tests ---\n")
df_base <- transform(data_2024_complete, grp = "baseline")
df_fu   <- transform(data_cohortB_baseline, grp = "cohortB")
combined <- rbind(df_base, df_fu)

fit_config <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE)
fit_metric <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = "loadings")
fit_scalar <- cfa(modelSyntax, combined, group = "grp", estimator = "MLR", std.lv = TRUE, meanstructure = TRUE, group.equal = c("loadings", "intercepts"))
compared_fit <- compareFit(fit_config, fit_metric, fit_scalar)
summary(compared_fit)






#### LONGTIUDINAL INVARIANCE TEST for COHORT A and B
library(semTools)

# ------------------------------------------------------------
# Reattach IDs and longitudinal metadata to the complete-case frames
# (the *_complete frames had study_submission_id stripped; the *_ids
#  vectors preserve row order, so we can put them back)
# ------------------------------------------------------------
d2024 <- data_2024_complete
d2024$study_submission_id <- as.character(data_2024_complete_ids)

d2026 <- data_2026_complete
d2026$study_submission_id <- as.character(ids_2026_complete)

attach_meta <- function(df) {
  idx <- match(df$study_submission_id, lookup$study_submission_id)
  df$longitudinal_id <- lookup$longitudinal_id[idx]
  df$wave            <- lookup$wave[idx]
  df
}
d2024 <- attach_meta(d2024)
d2026 <- attach_meta(d2026)

# Indicators used by modelSyntax (14 columns; connect_the_dots_I is shared by speed and EF but is a single column)
model_indicators <- c(
  "wordlist_correct_words","wordlist_delayed_correct_words","wordlist_recognition_correct_words",
  "semantic_fluency_score","phonemic_fluency_score","picture_naming_score",
  "avg_reaction_speed","fill_the_grid_total_time","clickskill_time","dragskill_time",
  "connect_the_dots_I_time_msec","connect_the_dots_II_time_msec",
  "digit_sequence_1_correct_series","digit_sequence_2_correct_series"
)

# Build a one-row-per-person wide frame from two long sources
build_wide <- function(t1_long, t2_long, indicators, id_col = "longitudinal_id") {
  t1 <- t1_long[, c(id_col, indicators)]
  t2 <- t2_long[, c(id_col, indicators)]
  t1 <- t1[!duplicated(t1[[id_col]]) & !is.na(t1[[id_col]]) & t1[[id_col]] != "", ]
  t2 <- t2[!duplicated(t2[[id_col]]) & !is.na(t2[[id_col]]) & t2[[id_col]] != "", ]
  names(t1)[-1] <- paste0(indicators, "_t1")
  names(t2)[-1] <- paste0(indicators, "_t2")
  merge(t1, t2, by = id_col)   # inner join keeps only matched persons
}

# ------------------------------------------------------------
# Cohort A wide frame: t1 = wave1 (2024), t2 = wave2 returning
# ------------------------------------------------------------
t2_A          <- d2026[is_wave2_returning, ]
returnA_long  <- unique(t2_A$longitudinal_id)
t1_A          <- subset(d2024, longitudinal_id %in% returnA_long)
wide_A        <- build_wide(t1_A, t2_A, model_indicators)
cat("Cohort A matched pairs:", nrow(wide_A), "\n")

# ------------------------------------------------------------
# Cohort B wide frame: t1 = wave2 occasion, t2 = wave3 occasion
# ------------------------------------------------------------
cohortB_long  <- unique(d2026$longitudinal_id[is_cohortB])
B_rows        <- subset(d2026, longitudinal_id %in% cohortB_long)
t1_B          <- subset(B_rows, wave == "wave2")
t2_B          <- subset(B_rows, wave == "wave3")
wide_B        <- build_wide(t1_B, t2_B, model_indicators)
cat("Cohort B matched pairs:", nrow(wide_B), "\n")


# Configural model: factors defined separately at t1 and t2,
long_model <- '
  memory_t1 =~ wordlist_correct_words_t1 + wordlist_delayed_correct_words_t1 + wordlist_recognition_correct_words_t1
  language_t1 =~ semantic_fluency_score_t1 + phonemic_fluency_score_t1 + picture_naming_score_t1
  speed_t1 =~ avg_reaction_speed_t1 + fill_the_grid_total_time_t1 + clickskill_time_t1 + dragskill_time_t1
  executive_function_t1 =~ connect_the_dots_I_time_msec_t1 + connect_the_dots_II_time_msec_t1 + digit_sequence_1_correct_series_t1 + digit_sequence_2_correct_series_t1
  speed_t1 =~ connect_the_dots_I_time_msec_t1
  digit_sequence_1_correct_series_t1 ~~ digit_sequence_2_correct_series_t1

  memory_t2 =~ wordlist_correct_words_t2 + wordlist_delayed_correct_words_t2 + wordlist_recognition_correct_words_t2
  language_t2 =~ semantic_fluency_score_t2 + phonemic_fluency_score_t2 + picture_naming_score_t2
  speed_t2 =~ avg_reaction_speed_t2 + fill_the_grid_total_time_t2 + clickskill_time_t2 + dragskill_time_t2
  executive_function_t2 =~ connect_the_dots_I_time_msec_t2 + connect_the_dots_II_time_msec_t2 + digit_sequence_1_correct_series_t2 + digit_sequence_2_correct_series_t2
  speed_t2 =~ connect_the_dots_I_time_msec_t2
  digit_sequence_1_correct_series_t2 ~~ digit_sequence_2_correct_series_t2
'

long_factor_names <- list(
  memory             = c("memory_t1", "memory_t2"),
  language           = c("language_t1", "language_t2"),
  speed              = c("speed_t1", "speed_t2"),
  executive_function = c("executive_function_t1", "executive_function_t2")
)


long_ind_names <- list(
  wordlist_correct_words             = c("wordlist_correct_words_t1",             "wordlist_correct_words_t2"),
  wordlist_delayed_correct_words     = c("wordlist_delayed_correct_words_t1",     "wordlist_delayed_correct_words_t2"),
  wordlist_recognition_correct_words = c("wordlist_recognition_correct_words_t1", "wordlist_recognition_correct_words_t2"),
  semantic_fluency_score             = c("semantic_fluency_score_t1",             "semantic_fluency_score_t2"),
  phonemic_fluency_score             = c("phonemic_fluency_score_t1",             "phonemic_fluency_score_t2"),
  picture_naming_score               = c("picture_naming_score_t1",               "picture_naming_score_t2"),
  avg_reaction_speed                 = c("avg_reaction_speed_t1",                 "avg_reaction_speed_t2"),
  fill_the_grid_total_time           = c("fill_the_grid_total_time_t1",           "fill_the_grid_total_time_t2"),
  clickskill_time                    = c("clickskill_time_t1",                    "clickskill_time_t2"),
  dragskill_time                     = c("dragskill_time_t1",                     "dragskill_time_t2"),
  connect_the_dots_I_time_msec       = c("connect_the_dots_I_time_msec_t1",       "connect_the_dots_I_time_msec_t2"),
  connect_the_dots_II_time_msec      = c("connect_the_dots_II_time_msec_t1",      "connect_the_dots_II_time_msec_t2"),
  digit_sequence_1_correct_series    = c("digit_sequence_1_correct_series_t1",    "digit_sequence_1_correct_series_t2"),
  digit_sequence_2_correct_series    = c("digit_sequence_2_correct_series_t1",    "digit_sequence_2_correct_series_t2")
)

fit_long <- function(wide_df, long.equal = character(0)) {
  measEq.syntax(
    configural.model  = long_model,
    data              = wide_df,
    ID.fac            = "std.lv",
    longFacNames = long_factor_names,
    longIndNames     = long_ind_names,
    long.equal        = long.equal,
    estimator         = "MLR",
    return.fit        = TRUE
  )
}

cat("\n================ Cohort A: wave1 -> wave2 (longitudinal) ================\n")
cfg    <- fit_long(wide_A)
metric <- fit_long(wide_A, "loadings")
scalar <- fit_long(wide_A, c("loadings", "intercepts"))
print(summary(compareFit(cfg, metric, scalar)))
invisible(list(configural = cfg, metric = metric, scalar = scalar))


cat("\n================ Cohort B: wave2 -> wave3 (longitudinal) ================\n")
cfg    <- fit_long(wide_B)
metric <- fit_long(wide_B, "loadings")
scalar <- fit_long(wide_B, c("loadings", "intercepts"))
print(summary(compareFit(cfg, metric, scalar)))in 
invisible(list(configural = cfg, metric = metric, scalar = scalar))


#### END LONGTIUDINAL INVARIANCE TEST for COHORT A and B




# Merge back to full participant list (including those with missing data)
factorScores <- merge(
  data.frame(study_submission_id = data_2026_list$study_submission_ids),
  factorScores,
  by = "study_submission_id",
  all.x = TRUE
)

# save
output_filename <- paste0(
  "/Users/jheitz/git/luha-prolific-study/src/resources/factor_scores_theory_2026_",
  format(Sys.time(), "%Y-%m-%d-%H%M"), ".csv"
)
#write.csv(factorScores, output_filename, row.names = FALSE)
cat("Saved to:", output_filename, "\n")




