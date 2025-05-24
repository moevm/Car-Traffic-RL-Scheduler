#!/bin/bash
export SUMO_HOME="/usr/local/share/sumo"

source sumovenv/bin/activate

tensorboard --logdir /app/ppo_traffic_lights_tensorboard --host=0.0.0.0 &

exec python3 main.py \
  -s configs/static_tls/cycle_time_10000/rand_40.sumocfg \
  -p configs/simulation_parameters/rand_40.json \
  -m train