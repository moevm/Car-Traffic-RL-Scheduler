#!/bin/bash

set -e

tensorboard --logdir /app/metrics_logs --host=0.0.0.0 &

python3 main.py \
  -s configs/static_tls/cycle_time_10000/rand_40.sumocfg \
  -p configs/simulation_parameters/rand_40.json \
  -m train

cp trained_model.zip pretrained_info/trained_model.zip
cp vec_normalized.pkl pretrained_info/vec_normalized.pkl

echo "Training finished."
echo "Model: pretrained_info/trained_model.zip"
echo "VecNormalize: pretrained_info/vec_normalized.pkl"