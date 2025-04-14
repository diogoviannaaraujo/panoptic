import runpod

print('Starting preloading...')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

print('Ended loading deps')

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

print('Ended loading model')