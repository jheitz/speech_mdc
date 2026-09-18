datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)


cd ..

# MAIN PAPER RESULTS
# CV only on Wave 1 and Cohort A data
#python -u run_multiple.py --config configs/regression/wave1_cohortA/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_wave1_cohortA
# Train on Cohort A, test on Cohort B
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB


# ABLATIONS: speech-only, demographics-only
#python -u run_multiple.py --config configs/regression/wave1_cohortA/svr_rbf_svr_all_individual_demographics_only_log_data.yaml --name svr_rbf_svr_all_individual_demographics_only_log_data --results_base_dir ${datetime_str}_svr_rbf_wave1_cohortA_demographics_only
#python -u run_multiple.py --config configs/regression/wave1_cohortA/svr_rbf_svr_all_individual_speech_only.yaml --name svr_rbf_svr_all_individual_speech_only --results_base_dir ${datetime_str}_svr_rbf_wave1_cohortA_speech_only
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual_demographics_only_log_data.yaml --name svr_rbf_svr_all_individual_demographics_only_log_data --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB_demographics_only
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual_speech_only.yaml --name svr_rbf_svr_all_individual_speech_only --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB_speech_only

# TRAIN LINGUISTICS&DEMOGRAPHICS MODEL TO APPLY TO PITT DATA
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual_linguistic_and_demographic_log_data.yaml --name svr_rbf_svr_all_individual_linguistic_and_demographics_log_data --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB_linguistic_and_demographics

# preprocess pitt to apply models to it
#python -u run.py --config configs/pitt/preprocess_pitt_linguistics_and_demographics.yaml --name preprocess_pitt_linguistics_and_demographics



#python -u run.py --config configs/test.yaml --name test
