#!/bin/bash
set -e

mlflow ui --host=0.0.0.0 --port=4272 &
MLFLOW_UI_PID=$!

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' &
JUPYTER_PID=$!

sleep 3

python train.py \
  --tracking-uri http://localhost:4272 \
  --epochs 1 \
  --batch-size 256 \
  --lr 0.001

wait $MLFLOW_UI_PID
wait $JUPYTER_PID
