#!/bin/bash
set -e

mlflow ui --host=0.0.0.0 --port=4272 &
MLFLOW_UI_PID=$!

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' &
JUPYTER_PID=$!

sleep 3

python train.py \
  --epochs  ${EPOCHS:-2} \
  --batch-size ${BATCH_SIZE:-128} \
  --lr         ${LR:-1e-3}

wait $MLFLOW_UI_PID
wait $JUPYTER_PID
