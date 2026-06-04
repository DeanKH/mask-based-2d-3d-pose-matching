FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    libopencv-dev \
    libglm-dev \
    libshaderc-dev \
    glslang-tools \
    libocct-foundation-dev \
    libocct-modeling-data-dev \
    libocct-modeling-algorithms-dev \
    libocct-data-exchange-dev \
    nlohmann-json3-dev \
    zlib1g-dev \
    libssl-dev \
    google-perftools \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY CMakeLists.txt /src/CMakeLists.txt
COPY mask-generation/ /src/mask-generation/
COPY src/ /src/src/

RUN mkdir -p /opt/build && cd /opt/build && cmake /src -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILER=ON && make -j$(nproc)

WORKDIR /workspace

ENTRYPOINT ["/opt/build/pose_matching"]
