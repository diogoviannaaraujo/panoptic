import runpod

print('Starting preloading...')

from transformers import Qwen2_5_VLForConditionalGeneration
import torch

print('Ended loading deps')

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print('Ended loading model')