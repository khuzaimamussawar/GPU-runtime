# syntax=docker/dockerfile:1.7
# Build FFmpeg ourselves: Ubuntu's packaged binary does not promise CUDA filters.
# CUDA 13 is baked into the image so Blackwell is a native build target. The pod
# provider still supplies the matching host kernel driver and NVENC/NVDEC device.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS ffmpeg-build

ARG FFMPEG_VERSION=7.1.1
ARG NV_CODEC_HEADERS_VERSION=n12.2.72.0

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git make g++ pkg-config yasm nasm xz-utils libgnutls28-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${NV_CODEC_HEADERS_VERSION} https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install PREFIX=/opt/ffmpeg \
    && rm -rf /tmp/nv-codec-headers

RUN curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && cd "/tmp/ffmpeg-${FFMPEG_VERSION}" \
    && PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig ./configure \
        --prefix=/opt/ffmpeg \
        --disable-debug \
        --disable-doc \
        --disable-shared \
        --enable-static \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-ffnvcodec \
        --enable-nvenc \
        --enable-nvdec \
        --enable-cuvid \
        --enable-libnpp \
        --extra-cflags=-I/opt/ffmpeg/include \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
    && make -j"$(nproc)" \
    && make install \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -filters | grep -q scale_cuda \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -q hevc_nvenc \
    && rm -rf "/tmp/ffmpeg-${FFMPEG_VERSION}" /tmp/ffmpeg.tar.xz

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RENDER_GPU_HOST=0.0.0.0 \
    RENDER_GPU_PORT=8000 \
    SCENEBUILDER_FFMPEG_FLAVOR=custom-cuda-ffmpeg-7.1.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates libgnutls30 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build /opt/ffmpeg /opt/ffmpeg
COPY render-fast/server.py /opt/scenebuilder-render/server.py

ENV PATH=/opt/ffmpeg/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /opt/scenebuilder-render
EXPOSE 8000
CMD ["python3", "server.py"]
