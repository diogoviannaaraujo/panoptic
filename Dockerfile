# Use Python 3.9 slim image as the base
FROM python:3.9-slim

# Install dependencies
RUN apt-get update; \ 
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git; \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Qwen 2.5 VL dependencies
RUN pip install git+https://github.com/huggingface/transformers accelerate

# Copy your handler code
COPY . .

# Run the handler code
RUN python -u preload.py

# Command to run when the container starts
CMD [ "python", "-u", "handler.py" ]
