
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl python3 python3-pip build-essential && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
# optional: FlashAttention (if compatible with your GPU/driver)
# RUN pip install flash-attn --no-build-isolation
COPY . .
