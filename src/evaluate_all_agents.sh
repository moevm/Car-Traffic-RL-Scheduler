#!/bin/bash

maps=("4" "20" "40" "80")
default_types=(
  "actuated_tls"
  "static_tls/cycle_time_45"
  "static_tls/cycle_time_60"
  "static_tls/cycle_time_90"
)
trained_type="static_tls/cycle_time_10000"

echo "[$(date)]"
for n in "${maps[@]}"; do
  for tls in "${default_types[@]}"; do
    for run in {1..5}; do
      echo -e "\n\nRunning $tls rand_$n default agent (run $run)"
      python3 main.py \
        -s "configs/${tls}/rand_${n}.sumocfg" \
        -p "configs/simulation_parameters/rand_${n}.json" \
        -m evaluation_default_agent
    done
  done

  for run in {1..5}; do
    echo -e "\n\nRunning $trained_type rand_$n trained agent (run $run)"
    python3 main.py \
      -s "configs/${trained_type}/rand_${n}.sumocfg" \
      -p "configs/simulation_parameters/rand_${n}.json" \
      -m evaluation_trained_agent
  done
done

echo "[$(date)]"

