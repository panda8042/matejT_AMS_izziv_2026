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
    numpy==1.26.4 \
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

# U-Mamba / Mamba dependencies
RUN pip install "causal-conv1d==1.2.0.post2" --no-build-isolation
RUN pip install "mamba-ssm==1.2.0.post1" --no-build-isolation
RUN pip install "transformers==4.39.3"

# Install U-Mamba from a fixed commit for reproducibility
RUN git clone https://github.com/bowang-lab/U-Mamba.git /opt/U-Mamba && \
    cd /opt/U-Mamba && \
    git checkout 28459e33ca03769800dd35e23c6e62491d1925b5

# Patch U-Mamba paths.py so it respects Docker environment variables
RUN python3 -c "from pathlib import Path; p=Path('/opt/U-Mamba/umamba/nnunetv2/paths.py'); t=p.read_text(); t=t.replace(\"nnUNet_raw = join(base, 'nnUNet_raw') # os.environ.get('nnUNet_raw')\", \"nnUNet_raw = os.environ.get('nnUNet_raw', join(base, 'nnUNet_raw'))\"); t=t.replace(\"nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')\", \"nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed', join(base, 'nnUNet_preprocessed'))\"); t=t.replace(\"nnUNet_results = join(base, 'nnUNet_results') # os.environ.get('nnUNet_results')\", \"nnUNet_results = os.environ.get('nnUNet_results', join(base, 'nnUNet_results'))\"); p.write_text(t)"

RUN pip install -e /opt/U-Mamba/umamba

# Force NumPy 1.x because PyTorch 2.0.1 / compiled extensions are not compatible with NumPy 2.x
RUN pip install "numpy==1.26.4"

# nnU-Net / U-Mamba paths inside the project directory
ENV nnUNet_raw=/workdir/nnUNet_raw
ENV nnUNet_preprocessed=/workdir/nnUNet_preprocessed
ENV nnUNet_results=/workdir/nnUNet_results

WORKDIR /workdir

CMD ["/bin/bash"]
