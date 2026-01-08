#!/bin/bash

# NOTE: Usage guide for run_reproducer_nan_xpu.sh
#
# This script is meant to reproduce NaN / non-finite issues on Aurora XPUs
# using previously saved debug snapshots (bad_actor_finite_*.pt files) and
# a Qwen2.5-0.5B-Instruct checkpoint. It expects to be run on an Aurora
# login/compute node with the frameworks module available.
#
# Basic usage:
#   1. Make sure you have a Python virtual environment with verl installed
#      (or rely on the default path in VERL_VENV_PATH below).
#   2. Make sure you have a clone of the verl repo containing this script
#      (or rely on the default path in VERL_REPO_PATH below).
#   3. Ensure that the debug snapshot files are present in the current
#      working directory:
#         - bad_actor_finite_input_*.pt
#         - bad_actor_finite_model_*.pt
#         - bad_actor_finite_optim_*.pt
#      The script will pick the first matching file for each pattern.
#   4. Make sure the Qwen2.5-0.5B-Instruct model is available under
#      DATA_MODEL_PATH (default: ./) at:
#         Qwen/Qwen2.5-0.5B-Instruct
#      i.e. the full path used will be:
#         ${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct
#   5. From the directory containing this script and the *.pt files, run:
#         chmod +x aurora/run_reproducer_nan_xpu.sh
#         ./aurora/run_reproducer_nan_xpu.sh
#
# Environment overrides:
#   - VERL_VENV_PATH: path to the Python virtualenv to activate
#   - VERL_REPO_PATH: path to the verl repo clone (used for PYTHONPATH)
#   - DATA_MODEL_PATH: base directory where the Qwen model is stored
#
# Example with custom paths:
#   export VERL_VENV_PATH=/path/to/venv
#   export VERL_REPO_PATH=/path/to/f_verl
#   export DATA_MODEL_PATH=/path/to/models
#   ./aurora/run_reproducer_nan_xpu.sh
#
# The script sets Intel oneCCL / XPU-related environment variables and then
# launches the reproducer with:
#   torchrun --nproc_per_node=2 reproducer_nan_xpu.py \
#       <BAD_INPUT> <BAD_MODEL> <BAD_OPTIM> <MODEL_PATH>
# where the four arguments are auto-populated as described above.

export VERL_VENV_PATH=${VERL_VENV_PATH:-./venv}
export VERL_REPO_PATH=${VERL_REPO_PATH:-./f_verl}
# path where model and data will be downloaded and preprocessed
export DATA_MODEL_PATH=${DATA_MODEL_PATH:=./}

# Load modules and activate environment
module load frameworks
source "${VERL_VENV_PATH}"/bin/activate

# Set PYTHONPATH to include the verl installation and reproducer script
echo "Setting PYTHONPATH and PATH"
export PYTHONPATH="${VERL_VENV_PATH}/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="${VERL_REPO_PATH}:$PYTHONPATH"

# CCL environment for Intel XPU backend
export CCL_ATL_TRANSPORT=mpi
export CCL_PROCESS_LAUNCHER=none
export CCL_OP_SYNC=0
export I_MPI_PIN_DOMAIN=auto
export I_MPI_PIN_ORDER=scatter
export CCL_LOCAL_RANK=0
export CCL_LOCAL_SIZE=1
export CCL_WORKER_COUNT=1
export ZE_AFFINITY_MASK=0

echo "CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT}"
echo "CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER}"
echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"

echo "Preparing reproducer arguments..."
# Expand the first matching timestamped files
BAD_INPUT=$(ls bad_actor_finite_input_*.pt | head -n 1)
BAD_MODEL=$(ls bad_actor_finite_model_*.pt | head -n 1)
BAD_OPTIM=$(ls bad_actor_finite_optim_*.pt | head -n 1)
MODEL_PATH="${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct"

echo "Using input: ${BAD_INPUT}"
echo "Using model: ${BAD_MODEL}"
echo "Using optimizer state: ${BAD_OPTIM}"
echo "Model path: ${MODEL_PATH}"

# Launch the reproducer script with torchrun for distributed FSDP on XPU
torchrun --nproc_per_node=2 reproducer_nan_xpu.py \
    ${BAD_INPUT} ${BAD_MODEL} ${BAD_OPTIM} ${MODEL_PATH}
