# syntax=docker/dockerfile:1.7
# Build FFmpeg ourselves so the selected NVENC encoders are always present.
# CUDA 13 is the runtime base for Blackwell. The pod provider still supplies
# the matching host kernel driver and NVENC device library.
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
        --enable-gnutls \
        --enable-ffnvcodec \
        --enable-nvenc \
        --extra-cflags=-I/opt/ffmpeg/include \
    && make -j"$(nproc)" \
    && make install \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -protocols | grep -qE '(^|[[:space:]])https([[:space:]]|$)' \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -q h264_nvenc \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -q hevc_nvenc \
    && /opt/ffmpeg/bin/ffprobe -hide_banner -version \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc2=s=16x16:r=25 -t 1 -vf fps=48 -pix_fmt yuv420p -f rawvideo /tmp/fps-check.yuv \
    && test "$(stat -c%s /tmp/fps-check.yuv)" -eq $((16 * 16 * 3 / 2 * 48)) \
    && rm -f /tmp/fps-check.yuv \
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
