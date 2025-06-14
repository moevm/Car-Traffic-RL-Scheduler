#!/bin/bash

tensorboard --logdir /app/metrics_logs --host=0.0.0.0 &

exec python3 main.py \
  -s configs/static_tls/cycle_time_10000/rand_40.sumocfg \
  -p configs/simulation_parameters/rand_40.json \
  -m train