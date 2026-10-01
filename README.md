# Code and data for "Spontaneous Speech as a Scalable Digital Biomarker for Monitoring Cognitive Change in Older Adults" 

This repository contains code and data for the paper submission "Spontaneous Speech as a Scalable Digital Biomarker for Monitoring Cognitive Change in Older Adults".

## Repository structure
- `conda/`: `environment_server.yaml` and `environment_local.yaml`: Specifying python and library versions for the server (for the data_analysis pipeline, which was run on a Linux server), and for a local setup (for the jupyter notebooks in `notebooks`, which were run on a MacBook computer)
- `src`: Code used for raw study data preparation and cross-sectional supervised regression.
- `src/data_preparation`: Code used in the preparation of the raw data from Prolific / ACS / study web server (e.g. automatic transcription of audio files, data quality checks)
- `src/data_analysis`: Data analysis pipeline code for data loading, feature extraction, regression models
- `notebooks/`: Jupyter notebooks for ad-hoc analyses and the analyses of results, including data statistics, hyperparameter testing, MDC calculation, and preparation of paper figures and tables
- `notebooks/transformed_data`: Composite cognitive scores and speech-based predictions for the four cognitive domains, the three speech tasks, and both longitudinal cohorts. This data is produced by `notebooks/evaluate_regression_and_transform_data.ipynb` based on results produced by the pipeline in `src/data_analysis`. This data allows the replication of most results,
including regression performance, test-retest reliabilities, and cognitive assessment and speech-based MDC estimates using the provided notebooks.


## Training and evaluating the machine learning regression models to predict cognitive composite scores from speech recordings
Before running, update the constants in `src/config/constants.py`. 
The main entry point for the analyses is `src/data_analysis/run/run.sh`. It runs the main pipeline using configurations available as `yaml` files in `src/data_analysis/configs`

Note: The code currently depends on the original data, which are not publicly available and cannot be shared in their entirety.
Contact the authors for access to a subset of participants, which can be granted in the context of a scientific collaboration.


## Calculating the Minimal Detectable Change (MDC) of speech-based predictions 
The core contribution of this publication, the MDC calculation for speech-based predictions, is implemented in `notebooks/MDC_calculation_and_SEM_analysis.ipynb`, based on the data in `notebooks/transformed_data`. 
