print('!! Starting...')

import os
import hashlib
import requests
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime

# torch.manual_seed(420)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", trust_remote_code=True)

print('!! Ended loading model')

def process_video_file(video_file):
    print(f"!! Processing video file: {video_file}")

    system_prompt = "You are an AI specialized in recognizing immediate danger to a person in videos scenes. Your mission is to analyze the video and generate the result in JSON format using structuire {description: 'small description of the scene', hasDanger: 'true if has danger in scene', dangerDescription: 'description of the danger if hasDanger is true"

    prompt = "QwenVL JSON "

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": video_file, "total_pixels": 40960 * 28 * 28, "min_pixels": 64 * 28 * 28, "fps": 30},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs["fps"]
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    print(datetime.now().strftime("%H:%M:%S"))

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
    print(output_text[0])
    print(datetime.now().strftime("%H:%M:%S"))

    # You might want to save the results to a file
    # with open(f"results_{os.path.basename(video_file)}.txt", "w") as f:
    #     f.write(output_text[0])

def handler():
    print("!! Starting process")

    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return

    # Get all video files in the dataset directory
    video_files = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ['.mp4']):
                video_files.append(os.path.join(root, file))

    video_files.sort()

    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return

    print(f"Found {len(video_files)} videos to process")

    for video_file in video_files:
        try:
            process_video_file(video_file)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue

handler()
