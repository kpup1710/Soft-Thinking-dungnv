# CUDA 12.4 + cuDNN 9 on Ubuntu 22.04, with Conda and Python 3.11.13
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV SHELL=/bin/bash

# System dependencies required by your pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge and create the conda env `st` with Python 3.11.13
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && conda config --system --set always_yes yes --set changeps1 no \
    && conda create -n st python=3.11.13 \
    && conda clean -afy

# Activate env by default
ENV CONDA_DEFAULT_ENV=st
ENV PATH=${CONDA_DIR}/envs/st/bin:${PATH}

# Ensure conda env is available in RUN commands
SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

# Copy pinned Python requirements (excluding local -e packages and torch stack)
COPY requirements.st.txt /tmp/requirements.st.txt

# Install pip tooling, PyTorch CUDA 12.4 stack, flashinfer, then the rest
RUN python -m pip install --upgrade "pip==25.1.1" "setuptools==78.1.1" "wheel==0.45.1" \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.6.0" "torchvision==0.21.0" "triton==3.2.0" \
    && python -m pip install --no-cache-dir \
        --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python \
        "flashinfer-python==0.2.3" \
    && python -m pip install --no-cache-dir -r /tmp/requirements.st.txt

# Install local editable packages
COPY LiveCodeBench_pkg /workspace/LiveCodeBench_pkg
RUN cd /workspace/LiveCodeBench_pkg \
    && python -m pip install -e .

COPY sglang_soft_thinking_pkg/python /workspace/sglang
RUN cd /workspace/sglang \
    && python -m pip install -e .

CMD ["bash"]
