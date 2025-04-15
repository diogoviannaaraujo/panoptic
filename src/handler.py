import runpod

print('!! Starting server...')

import os
import hashlib
import requests
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print('!! Ended loading deps')

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./qwen-vl-32b-awq",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("./qwen-vl-32b-awq", trust_remote_code=True)

print('!! Ended loading model')

def download_video(url):
    os.makedirs(".cache", exist_ok=True)
    video_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    response = requests.get(url, stream=True)
    file_path = f".cache/{video_hash}.mp4"
    if os.path.exists(file_path):
        print(f"Video already exists at {file_path}")
        return file_path
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {file_path}")
    return file_path


def handler(event):
    print("!! Starting handler")

    input = event['input']
    instruction = input.get('instruction')
    video_url = input.get('url')

    video_file = download_video(video_url)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_file, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    print(f"!! Starting inference in {video_file}")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs["fps"]
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

