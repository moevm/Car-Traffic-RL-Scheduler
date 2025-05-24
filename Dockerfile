FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

LABEL Description="Containerized traffic RL-scheduler in SUMO"

ENV SUMO_VERSION 1.21.0
ENV SUMO_MAKE_FOLDER /opt/sumo

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
    wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz && \
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

ENV VIRTUAL_ENV sumovenv

COPY ./src/requirements.txt /app/requirements.txt

RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install -r requirements.txt --no-cache-dir && \
    apt-get clean

COPY ./src /app
COPY entrypoint.sh /app/entrypoint.sh

RUN groupadd -r user && useradd -g user user
RUN chown -R user:user /app
USER user

CMD ["./entrypoint.sh"]