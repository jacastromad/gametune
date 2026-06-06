FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mido" \
    "pytest" \
    "pyyaml"

