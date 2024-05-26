# a script to run the precomputation as slurm jobs on the cluster
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <game_name> <config_id>"
    exit 1
fi

# name of the python file to run
PYTHON_FILE="pre_compute.py"

# command line arguments
game_name=$1
config_id=$2
NAME=${game_name}_${config_id}

# slurm settings
MEMORY=4096

# Paths and directories
PROJECT_DIR=$(pwd)
SCRIPT_DIR=${PROJECT_DIR}/benchmark
echo "Project directory: ${PROJECT_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p ${LOG_DIR}
echo "Log directory: ${LOG_DIR}"
ERROR_DIR=${PROJECT_DIR}/errors
mkdir -p ${ERROR_DIR}
echo "Error directory: ${ERROR_DIR}"

# create the slurm command file
FILE="run_${NAME}.cmd"
echo "${FILE}"

# fill the slurm command file with slurm settings
echo "#!/bin/bash" > "${FILE}"
echo "#SBATCH -J ${NAME}" >> "${FILE}"
echo "#SBATCH -o ${LOG_DIR}/${NAME}.log" >> "${FILE}"
echo "#SBATCH -e ${ERROR_DIR}/${NAME}.err" >> "${FILE}"
echo "#SBATCH --get-user-env" >> "${FILE}"
echo "#SBATCH --mem=${MEMORY}" >> "${FILE}"
echo "#SBATCH --time 23:50:00" >> "${FILE}"
echo "#SBATCH --mail-user=Maximilian.Muschalik@ifi.lmu.de" >> "${FILE}"
echo "#SBATCH --mail-type=NONE" >> "${FILE}"
echo "#SBATCH --cpus-per-task=1" >> "${FILE}"

# source the virtual environment
echo "source ${PROJECT_DIR}/venv/bin/activate" >> "${FILE}"

# add the command to run the python file
echo "python ${SCRIPT_DIR}/${PYTHON_FILE} ${game_name} ${config_id}" >> "${FILE}"
echo "deactivate" >> "${FILE}"

# submit the job
sbatch "${FILE}"
echo "Submitted job ${NAME}"
