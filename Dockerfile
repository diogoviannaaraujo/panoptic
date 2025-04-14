ARG CUDA_VERSION=12.6.3
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
&& rm -rf /var/lib/apt/lists/*
EOF

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# install torch
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install flash attention
RUN pip3 install --no-build-isolation flash-attn==2.7.4.post1

# Copy your handler code
COPY ./src .

# Run the handler code
RUN python3 -u preload.py

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Command to run when the container starts
CMD [ "python3", "-u", "handler.py" ]
