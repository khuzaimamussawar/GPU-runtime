# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RENDER_GPU_HOST=0.0.0.0 \
    RENDER_GPU_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY render-fast/server.py /opt/scenebuilder-render/server.py

WORKDIR /opt/scenebuilder-render
EXPOSE 8000
CMD ["python3", "server.py"]
