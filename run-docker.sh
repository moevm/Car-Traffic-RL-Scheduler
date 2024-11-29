docker build . -t docker-sumo
xhost +local:docker
docker run -it --rm \
    --env="$DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="./src/configs:/app/configs" \
    --device /dev/dri \
    docker-sumo