# sh file to run multiple python scripts in parallel on the cluster for the pre_compute.py file.
# Note: This file does not run with SLURM but just with the bash shell.
# Path: benchmark/pre_compute.sh
# on the cluster, run this script with the following command:
# chmod +x pre_compute.sh
# ./pre_compute.sh

# name of the python file to run
PYTHON_FILE="pre_compute.py"

# script settings: PYTHON_FILE <game_name> <config_id> <n_player_id>

# run the python file with nohup to run in the background
nohup python $PYTHON_FILE --game AdultCensusRandomForestEnsembleSelection --config_id 1 &
nohup python $PYTHON_FILE --game BikeSharingRandomForestEnsembleSelection --config_id 1 &
nohup python $PYTHON_FILE --game CaliforniaHousingRandomForestEnsembleSelection --config_id 1 &
nohup python $PYTHON_FILE --game ImageClassifierLocalXAI --config_id 1 --n_player_id 2 &
nohup python $PYTHON_FILE --game ImageClassifierLocalXAI --config_id 1 --n_player_id 3 &
nohup python $PYTHON_FILE --game SentimentAnalysisLocalXAI --config_id 1 --n_player_id 1 &
nohup python $PYTHON_FILE --game AdultCensusFeatureSelection --config_id 1 &
nohup python $PYTHON_FILE --game AdultCensusFeatureSelection --config_id 2 &
nohup python $PYTHON_FILE --game AdultCensusFeatureSelection --config_id 3 &
nohup python $PYTHON_FILE --game BikeSharingFeatureSelection --config_id 1 &
nohup python $PYTHON_FILE --game BikeSharingFeatureSelection --config_id 2 &
nohup python $PYTHON_FILE --game BikeSharingFeatureSelection --config_id 3 &
nohup python $PYTHON_FILE --game CaliforniaHousingFeatureSelection --config_id 1 &
nohup python $PYTHON_FILE --game CaliforniaHousingFeatureSelection --config_id 2 &
nohup python $PYTHON_FILE --game CaliforniaHousingFeatureSelection --config_id 3 &
nohup python $PYTHON_FILE --game AdultCensusEnsembleSelection --config_id 1 &
nohup python $PYTHON_FILE --game BikeSharingEnsembleSelection --config_id 1 &
nohup python $PYTHON_FILE --game CaliforniaHousingEnsembleSelection --config_id 1 &
