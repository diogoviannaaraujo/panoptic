#!/bin/bash

apt install python3-pip python3-venv

python3 -m venv .env

.env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.env/bin/pip install transformers==4.52.4 "qwen-vl-utils[decord]" autoawq accelerate

