import runpod
import time

print('!! Starting server...')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

print('!! Ended loading deps')

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

print('!! Ended loading model')

def handler(event):
    print("!! Starting handler")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    print("!! Starting inference")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    print("!! Starting inference generation")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("!! Ended inference")
    print(output_text)


    input = event['input']
    instruction = input.get('instruction')
    seconds = input.get('seconds', 0)

    # Placeholder for a task; replace with image or text generation logic as needed
    time.sleep(seconds)
    result = instruction.replace(instruction.split()[0], 'created', 1)

    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
