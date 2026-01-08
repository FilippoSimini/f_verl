#!/bin/bash -l
# from: https://verl.readthedocs.io/en/latest/start/quickstart.html

export VERL_VENV_PATH=${VERL_VENV_PATH:-./venv}
export VERL_REPO_PATH=${VERL_REPO_PATH:-./f_verl}
# path where model and data will be downloaded and preprocessed
export DATA_MODEL_PATH=${DATA_MODEL_PATH:=./}

export NOW=$(date +%Y-%m-%d_%H%M)
export LOG_FILE=verl_demo_ccl_fixed_v2_${NOW}.log

# Create log file and append script content
echo -e "=== Start of $(realpath $0)  ===\n" > ${LOG_FILE}
cat "$0" | envsubst  >> ${LOG_FILE}
echo -e "\n=== End of $(realpath $0)  ===\n" >> ${LOG_FILE}

# Load modules and activate environment
module load frameworks
source "${VERL_VENV_PATH}"/bin/activate
#source ~/ccl.env
echo "CCL root: " ${CCL_ROOT} | tee -a ${LOG_FILE}

# Set PYTHONPATH to include the verl installation
export PYTHONPATH="${VERL_VENV_PATH}/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="${VERL_REPO_PATH}:$PYTHONPATH"

echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOG_FILE}
echo "Python executable: $(which python3)" | tee -a ${LOG_FILE}

# Set CCL environment variables to avoid PMIx issues
# Force CCL to use MPI instead of PMIx which is failing
export CCL_ATL_TRANSPORT=mpi
export CCL_PROCESS_LAUNCHER=none  # Disable automatic launcher detection
export CCL_OP_SYNC=0  # Use default value instead of forced 1
export I_MPI_PIN_DOMAIN=auto
export I_MPI_PIN_ORDER=scatter
# Additional CCL settings for single node
export CCL_LOCAL_RANK=0
export CCL_LOCAL_SIZE=1
export CCL_WORKER_COUNT=1
# Set ZE_AFFINITY_MASK for Intel GPU
export ZE_AFFINITY_MASK=0

echo "CCL Settings:" | tee -a ${LOG_FILE}
echo "  CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT}" | tee -a ${LOG_FILE}
echo "  CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER}" | tee -a ${LOG_FILE}
echo "  CCL_LOCAL_RANK=${CCL_LOCAL_RANK}" | tee -a ${LOG_FILE}
echo "  CCL_LOCAL_SIZE=${CCL_LOCAL_SIZE}" | tee -a ${LOG_FILE}
echo "  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}" | tee -a ${LOG_FILE}

# Verify verl can be imported before starting Ray
python3 -c "import verl; print(f'VERL imported successfully from: {verl.__file__}')" | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot import verl module. Exiting." | tee -a ${LOG_FILE}
    exit 1
fi

# Start ray
export RAY_TMPDIR="/tmp/raytmp"
mkdir -p $RAY_TMPDIR
export HSN_IP_ADDRESS=$(getent hosts "$(hostname).hsn.cm.aurora.alcf.anl.gov" | awk '{ print $1 }' | sort | head -n 1)
ray start --num-gpus=1 --num-cpus=2 --head --node-ip-address="$HSN_IP_ADDRESS" --temp-dir=/tmp
# Wait for Ray to fully initialize
sleep 5

# Test Ray cluster and verl import
python3 -c "
import ray
import os
print('Testing Ray and verl import...')
ray.init(address='auto')

@ray.remote
def test_verl_import():
    import sys
    import os
    print(f'Worker PYTHONPATH: {sys.path}')
    print(f'Worker CCL_ATL_TRANSPORT: {os.environ.get(\"CCL_ATL_TRANSPORT\", \"not set\")}')
    try:
        import verl
        return f'SUCCESS: verl imported in worker from {verl.__file__}'
    except ImportError as e:
        return f'FAILED: {e}'

result = ray.get(test_verl_import.remote())
print(result)
ray.shutdown()
" | tee -a ${LOG_FILE}


# downlaod data and model
mkdir -p "${DATA_MODEL_PATH}"

# gsm8k dataset
if [ -f "${DATA_MODEL_PATH}/openai/gsm8k/train.parquet" ]; then
  echo "gsm8k dataset already present at ${DATA_MODEL_PATH}/openai/gsm8k, skipping clone"
else
  git clone https://huggingface.co/datasets/openai/gsm8k "${DATA_MODEL_PATH}/openai/gsm8k"
  wget -O "${DATA_MODEL_PATH}/openai/gsm8k/gsm8k.py" https://raw.githubusercontent.com/volcengine/verl/main/examples/data_preprocess/gsm8k.py
  python3 "${DATA_MODEL_PATH}/openai/gsm8k/gsm8k.py" --local_save_dir "${DATA_MODEL_PATH}/openai/gsm8k"
fi

# Qwen model
if [ -d "${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct" ]; then
  echo "Qwen2.5-0.5B-Instruct already present at ${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct, skipping clone"
else
  git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct "${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct"
fi

export PRECISION=bf16
#export PRECISION=fp32

# Run the main training script with environment variables passed through Ray
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
 data.train_files=${DATA_MODEL_PATH}/openai/gsm8k/train.parquet \
 data.val_files=${DATA_MODEL_PATH}/openai/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.fsdp_config.model_dtype=${PRECISION} \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.free_cache_engine=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.actor.use_torch_compile=False \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.fsdp_config.model_dtype=${PRECISION} \
 critic.optim.lr=1e-5 \
 critic.model.path=${DATA_MODEL_PATH}/Qwen/Qwen2.5-0.5B-Instruct \
 critic.model.fsdp_config.model_dtype=${PRECISION} \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 +ray_kwargs.ray_init.address="auto" \
 trainer.default_local_dir=/tmp/ \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_training_steps=3 2>&1 | tee -a ${LOG_FILE}

# actor_rollout_ref.model.attn_implementation=eager \

ray stop
