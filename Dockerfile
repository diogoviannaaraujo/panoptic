ARG CUDA_VERSION=12.1.0
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

FROM ${from} as base

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

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash attention
# RUN pip3 install --no-build-isolation flash-attn==2.7.2.post1

# Copy your handler code
COPY ./src .

# Run the handler code
RUN python -u preload.py

# Command to run when the container starts
CMD [ "python", "-u", "handler.py" ]
