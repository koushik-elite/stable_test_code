import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers.utils import make_image_grid, load_image
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image

device = "cuda"
# model_id_or_path = "acheong08/f222"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
# model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

# pipe = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(device)
# pipe.enable_model_cpu_offload()

init_image = Image.open("assets/myface/lax2.png").convert("RGB")
width, height = init_image.size
ratio = width / height
new_height = 900
new_width = int(ratio * new_height)
print(new_width, new_height)
init_image = init_image.resize((new_width, new_height))
init_image.save(f"output/init_images_resize.png")
# prompt = "An astronaut riding a green horse"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
# init_image = load_image(url)

prompt = "Astronaut in a war zone, guns in hands, detailed, 8k, best quality, high quality"
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, free hair, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin"

all_images = pipe(prompt=prompt, negative_prompt=neg_prompt, image=init_image, num_inference_steps=50, strength=0.8, guidance_scale=12).images

# print(all_images)
n = 0
for image in all_images:
    n = n + 1
    # print(image)
    image.save(f"output/images_{n}.png")
