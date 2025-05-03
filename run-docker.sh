docker build . -t docker-sumo
docker run -it --rm \
    --env DISPLAY=$DISPLAY \
    -u $(id -u):$(id -g) \
    --env XAUTHORITY=$XAUTHORITY \
    --volume $XAUTHORITY:$XAUTHORITY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume="./src/configs:/app/configs" \
    --volume="./src/ppo_traffic_lights_tensorboard:/app/ppo_traffic_lights_tensorboard" \
    --volume="./src/pre_training:/app/pre_training" \
    --device /dev/dri \
    -p 6006:6006 \
    --gpus all \
    docker-sumo