
FROM vogoid/ai-dev

# Clone model
RUN huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct-AWQ --local-dir ./qwen-vl-32b-awq

# List the contents of the directory
RUN ls ./qwen-vl-32b-awq

# Copy your handler code
COPY ./src .

# Command to run when the container starts
CMD [ "python3", "-u", "handler.py" ]
