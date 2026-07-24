datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)


cd ..

# MAIN PAPER RESULTS
# CV only on Wave 1 and Cohort A data
#python -u run_multiple.py --config configs/regression/wave1_cohortA/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_wave1_cohortA
# Train on Cohort A, test on Cohort B
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB

# TRANSFORMER-BASED APPROACH: combine fine-tuned features from wavlm and deberta, using generalized Ridge regression
# python -u run.py --script_path scripts/generalizedRidge_combine_wavlm_deberta.py --config configs/transformer_based/generalizedRidge_combine_wavlm_deberta/combine_language_v2.yaml --name combine_wavlm_deberta_language_v2


#python -u run.py --config configs/test.yaml --name test
