#!/bin/bash

docker build . -t docker-sumo
mkdir -p src/metrics_logs/ src/pretrained_info/ src/statistics/
docker run -it --rm \
  --env DISPLAY=$DISPLAY \
  --env XAUTHORITY=$XAUTHORITY \
  --volume $XAUTHORITY:$XAUTHORITY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume="./src/configs:/app/configs" \
  --volume="./src/metrics_logs:/app/metrics_logs" \
  --volume="./src/pretrained_info:/app/pretrained_info" \
  --volume="./src/statistics:/app/statistics" \
  --device /dev/dri \
  -p 6006:6006 \
  --gpus all \
  docker-sumo "$@"
