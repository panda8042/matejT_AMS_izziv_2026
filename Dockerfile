FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workdir

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-image \
    scikit-learn \
    nibabel \
    SimpleITK \
    tqdm \
    einops \
    timm==0.5.4 \
    packaging \
    connected-components-3d

WORKDIR /workdir

CMD ["/bin/bash"]
