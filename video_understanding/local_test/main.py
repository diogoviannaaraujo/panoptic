print('!! Starting server...')

import os
import hashlib
import requests
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime

print('!! Ended loading deps')

torch.manual_seed(420)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", trust_remote_code=True)

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


def handler():
    print("!! Starting handler")

    system_prompt = "You are an AI specialized in recognizing immediate danger to a person in videos scenes. Your mission is to analyze the video and generate the result in JSON format using structuire {description: 'small description of the scene', hasDanger: 'true if has danger in scene', dangerDescription: 'description of the danger if hasDanger is true'}."

    prompt = "QwenVL JSON "

    video_url = ""
#    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4" # Package demo
#    video_url = "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4" # Garage truck tight street
#    video_url = "https://ia902303.us.archive.org/30/items/1_20210928_20210928_1312/1.ia.mp4"
    video_url = "https://archive.org/download/youtube-VChIjKSoX6Y/VChIjKSoX6Y.mp4" # Garage fight
#    video_url = "https://archive.org/download/clvmn-Lakeville_Bank_Robbery_Case_18004636/Lakeville_Bank_Robbery_Case_18004636.mp4" # Bank robbery

    video_file = download_video(video_url)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": video_file, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28},
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
    print("images", image_inputs)
    #print("videos", video_inputs.length)
    print("text", text)
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

handler()
