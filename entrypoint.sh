#!/bin/bash

set -e

export SUMO_HOME="/usr/local/share/sumo"
source sumovenv/bin/activate

SCRIPT="$1"
shift

if [[ -z "$SCRIPT" ]]; then
  echo "No script specified. Use: train.sh or evaluate.sh."
  exit 1
fi

if [[ ! -x "$SCRIPT" ]]; then
  echo "Script $SCRIPT not found or not executable."
  exit 1
fi

echo "Running: $SCRIPT $@"
exec "./$SCRIPT" "$@"
