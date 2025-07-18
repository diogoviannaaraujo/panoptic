from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import os

# Load the model

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True, # Uncomment for GPU acceleration & pip install accelerate 
    device_map="cuda:0"   
)

# Process all images in the images directory
image_dir = "./images"

prompt = "Is there any immediate danger in the image? Answer in JSON using the format {sceneDescription: string, danger: bool, dangerDescription: string}"

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        
        print(f"\nProcessing {image_file}:")
        start_time = time.perf_counter()
        result = model.query(image, prompt)["answer"]
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(result)
        
