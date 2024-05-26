# sh file to run multiple python scripts in parallel on the cluster for the pre_compute.py file.
# Note: This file does not run with SLURM but just with the bash shell.
# Path: benchmark/pre_compute.sh
# on the cluster, run this script with the following command:
# chmod +x pre_compute.sh
# ./pre_compute.sh

# name of the python file to run
PYTHON_FILE="pre_compute.py"

# script settings: PYTHON_FILE <game_name> <config_id>

# run the python file
python $PYTHON_FILE AdultCensusUnsupervisedData 1 &
python $PYTHON_FILE AdultCensusClusterExplanation 1 &
python $PYTHON_FILE AdultCensusClusterExplanation 2 &
python $PYTHON_FILE BikeSharingClusterExplanation 1 &
python $PYTHON_FILE BikeSharingClusterExplanation 2 &
python $PYTHON_FILE CaliforniaHousingClusterExplanation 1 &
python $PYTHON_FILE CaliforniaHousingClusterExplanation 2 &
python $PYTHON_FILE AdultCensusEnsembleSelection 1 &
python $PYTHON_FILE BikeSharingEnsembleSelection 1 &
python $PYTHON_FILE CaliforniaHousingEnsembleSelection 1 &
python $PYTHON_FILE AdultCensusRandomForestEnsembleSelection 1 &
python $PYTHON_FILE BikeSharingRandomForestEnsembleSelection 1 &
python $PYTHON_FILE CaliforniaHousingRandomForestEnsembleSelection 1 &
