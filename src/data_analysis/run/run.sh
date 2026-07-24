datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)


cd ..

# MAIN PAPER RESULTS
# CV only on Wave 1 and Cohort A data
#python -u run_multiple.py --config configs/regression/wave1_cohortA/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_wave1_cohortA
# Train on Cohort A, test on Cohort B
#python -u run_multiple.py --config configs/regression/test_on_cohortB/svr_rbf_svr_all_individual.yaml --name svr_rbf_svr_all_individual --results_base_dir ${datetime_str}_svr_rbf_test_on_cohortB

#python -u run.py --config configs/test.yaml --name test
