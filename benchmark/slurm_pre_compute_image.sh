# a script to run the precomputation as slurm jobs on the cluster
# on the cluster, run this script with the following command:
# chmod +x slurm_run.sh
# ./slurm_run.sh <game_name> <config_id>

# name of the python file to run
PYTHON_FILE="pre_compute_image.py"

NAME="pre_compute_image_games"

# slurm settings
MEMORY=4096

# Paths and directories
# get the project directory
PROJECT_DIR=$(git rev-parse --show-toplevel)
echo "Project directory: ${PROJECT_DIR}"
SCRIPT_DIR=${PROJECT_DIR}/benchmark
echo "Script directory: ${SCRIPT_DIR}"
LOG_DIR=${SCRIPT_DIR}/logs
mkdir -p ${LOG_DIR}
echo "Log directory: ${LOG_DIR}"
ERROR_DIR=${SCRIPT_DIR}/errors
mkdir -p ${ERROR_DIR}
echo "Error directory: ${ERROR_DIR}"

# create the slurm command file
FILE="${SCRIPT_DIR}/${NAME}.cmd"
echo "${FILE}"

# fill the slurm command file with slurm settings
echo "#!/bin/bash" >> "${FILE}"
echo "#SBATCH -J ${NAME}" >> "${FILE}"
echo "#SBATCH -o ${LOG_DIR}/${NAME}.log" >> "${FILE}"
echo "#SBATCH -e ${ERROR_DIR}/${NAME}.err" >> "${FILE}"
echo "#SBATCH --get-user-env" >> "${FILE}"
echo "#SBATCH --mem=${MEMORY}" >> "${FILE}"
echo "#SBATCH --time 23:50:00" >> "${FILE}"
echo "#SBATCH --mail-user=Maximilian.Muschalik@ifi.lmu.de" >> "${FILE}"
echo "#SBATCH --mail-type=NONE" >> "${FILE}"
echo "#SBATCH --cpus-per-task=1" >> "${FILE}"
echo "#SBATCH --qos=bg_maximilian" >> "${FILE}"
echo "#SBATCH --nodeslist=cmp01-kiml" >> "${FILE}"

# source the virtual environment
echo "source ${PROJECT_DIR}/venv/bin/activate" >> "${FILE}"

# add the command to run the python file
echo "python ${SCRIPT_DIR}/${PYTHON_FILE}" >> "${FILE}"
echo "deactivate" >> "${FILE}"

# submit the job
sbatch "${FILE}"
echo "Submitted job ${NAME}"
