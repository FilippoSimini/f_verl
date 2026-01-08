#!/bin/bash

export WORKING_DIR=$(realpath)
export VERL_VENV_PATH=${WORKING_DIR}/venv
export VERL_REPO_PATH=${WORKING_DIR}/f_verl

module load frameworks

# test if verl venv exists in current directory and works
if [ -d "${VERL_VENV_PATH}" ]; then
  source "${VERL_VENV_PATH}/bin/activate"
  python -c "import verl" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "Existing verl venv found but verl cannot be imported. Recreating venv and reinstalling verl..."
    deactivate 2>/dev/null || true
    rm -rf "${VERL_VENV_PATH}"
  else
    echo "Existing verl venv found and verl can be imported. Skipping installation."
  fi
fi

# if venv directory does not exist (or was removed), create venv and install verl
if [ ! -d "${VERL_VENV_PATH}" ]; then
  echo "verl venv not found or invalid in the current directory. Creating it..."
  python3 -m venv --system-site-packages venv
  source "${VERL_VENV_PATH}/bin/activate"
  if [ ! -d "${VERL_REPO_PATH}" ]; then
    git clone https://github.com/FilippoSimini/f_verl.git
  fi
  cd "${VERL_REPO_PATH}"
  pip install -e .
fi


# run verl
git switch debug
echo "Running ${VERL_REPO_PATH}/aurora/verl_demo_xpu.sh"
echo "================================================="
${VERL_REPO_PATH}/aurora/verl_demo_xpu.sh

# run reproducer
echo "Running nan reporducer ${VERL_REPO_PATH}/aurora/run_reproducer_nan_xpu.sh"
echo "========================================================================="
${VERL_REPO_PATH}/aurora/run_reproducer_nan_xpu.sh

