#!/bin/bash
export SUMO_HOME="/opt/sumo"

source sumovenv/bin/activate

tensorboard --logdir /app/ppo_traffic_lights_tensorboard --host=0.0.0.0 &

exec python3 main.py \
  -s configs/learning-configs/learning_small.sumocfg \
  -p configs/learning-configs/learning_parameters.json \
  -m train