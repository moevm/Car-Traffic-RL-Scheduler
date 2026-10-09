FROM nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04

LABEL Description="Containerized traffic RL-scheduler in SUMO"

ENV SUMO_VERSION=1.22.0 SUMO_MAKE_FOLDER=/opt/sumo

RUN apt-get update && \
    apt-get install -y cmake \
    python3 \
    wget \
    g++ \
    libxerces-c-dev \
    libfox-1.6-dev \
    libgdal-dev \
    libproj-dev \
    libgl2ps-dev \
    python3-dev \
    swig \
    default-jdk \
    maven \
    libeigen3-dev && \
    wget https://sumo.dlr.de/releases/$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz && \
    tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_MAKE_FOLDER && \
    rm sumo-src-$SUMO_VERSION.tar.gz && \
    cd $SUMO_MAKE_FOLDER && \
    export SUMO_HOME="$SUMO_MAKE_FOLDER" && \
    cmake -B build . && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    apt-get install -y python3.12 \
    python3-pip \
    python3.12-venv && \
    apt-get clean

WORKDIR /app

ENV VIRTUAL_ENV=sumovenv

COPY ./src/requirements.txt /app/requirements.txt

RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --timeout 3600 --retries 10 \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-cache-dir

RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --timeout 3600 --retries 10 \
        -r requirements.txt \
        --no-cache-dir
COPY ./src /app
COPY *.sh /app
RUN groupadd -r -g 1001 user && \
    useradd -m -u 1001 -g 1001 user
RUN chown -R 1001:1001 /app
RUN chmod +x /app/entrypoint.sh /app/train.sh /app/evaluate.sh /app/evaluate_all_agents.sh
USER 1001:1001

ENTRYPOINT ["./entrypoint.sh"]
CMD ["./train.sh"]
