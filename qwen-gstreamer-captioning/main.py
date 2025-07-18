
#!/usr/bin/env python3
"""webcam_qwen_pipeline.py
---------------------------------
End‑to‑end demo:

* Capture webcam via **GStreamer**
* Split into 5‑second MP4 clips (GPU H.264 if available)
* After each clip closes, run **Qwen‑VL‑7B** (CUDA) to caption
* Append to log:  <unix‑timestamp>.mp4    <caption>

Dependencies
------------
sudo apt install -y python3-gi gstreamer1.0-tools gstreamer1.0-libav \
                   gstreamer1.0-plugins-{base,good,bad,ugly}
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow opencv-python

Usage
-----
python webcam_qwen_pipeline.py --device /dev/video0 --out clips --log captions.log

"""

import argparse, time, sys
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# -------------------------- CLI ----------------------------
p = argparse.ArgumentParser()
p.add_argument('--device', default='/dev/video0', help='v4l2 device path')
p.add_argument('--width', type=int, default=640)
p.add_argument('--height', type=int, default=480)
p.add_argument('--fps', type=int, default=15)
p.add_argument('--out', default='clips')
p.add_argument('--log', default='captions.log')
p.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
args = p.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

print('Loading Qwen‑2.5‑VL‑7B …')
processor = AutoProcessor.from_pretrained(args.model)
llm = LLM(
    model=args.model,
    max_model_len=32768 if smart_resize is None else 4096,
    max_num_seqs=5,
    limit_mm_per_prompt={"video": 1},
)
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)
print('Model loaded')

# ------------------- Caption helper -----------------------
def caption_clip(path: Path) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
                {
                    "type": "video", 
                    "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                }
            ]
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,

        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text

# ----------------- Build GStreamer pipe -------------------
Gst.init(None)
enc = ('nvh264enc preset=4 bitrate=2000' if Gst.ElementFactory.find('nvh264enc')
       else 'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast')
pipe_desc = f"""
    uridecodebin uri="udp://127.0.0.1:5000" !
    splitmuxsink name=smux max-size-time=5000000000
"""
pipeline = Gst.parse_launch(pipe_desc)
smux = pipeline.get_by_name('smux')

# Filename callback
def fmt_location(sink, fragment_id):
    return f"{args.out}/{int(time.time())}.mp4"
smux.connect('format-location', fmt_location)

# Bus handler: caption closed fragments
def on_bus(bus, msg):
    if msg.type == Gst.MessageType.ELEMENT and msg.has_name('splitmuxsink-fragment-closed'):
        fname = msg.get_structure().get_string('location')
        if fname:
            try:
                caption = caption_clip(Path(fname))
                with open(args.log, 'a') as f:
                    f.write(f"{Path(fname).name}\t{caption}\n")
                print(f"Captioned {Path(fname).name} → {caption}")
            except Exception as e:
                print('Error captioning', fname, e, file=sys.stderr)
    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect('message', on_bus)

# -------------------- Run loop ----------------------------
print('Capturing…  Ctrl+C to stop')
pipeline.set_state(Gst.State.PLAYING)
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
    print('Stopped.')
