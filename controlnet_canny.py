from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
from tqdm import tqdm

import itertools
import math
import os.path
import random

num_images_per_prompt: int = 1
num_inference_steps: int = 20
strengths = list(range(3, 10))
conditioning_scale = list(range(1, 10))
guidance_scales = list(range(2, 17))
eta_list = list(range(0, 11))
size_factor: float = 0.99
combined_list = list(itertools.product(conditioning_scale, strengths, guidance_scales))
generator = torch.Generator(device="cuda").manual_seed(2257817932)


# controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16,)
pipe.enable_model_cpu_offload()

image = Image.open("assets/images/sketch_merge.jpg").convert("RGB")
width, height = image.size
ratio = width / height
new_height = 900
new_width = int(ratio * new_height)
print(new_width, new_height)
image = image.resize((new_width, new_height))

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

prompt = "Human Infographic image, photo realistic, Ultra realistic, high quality, extremely detailed. all on Stable Diffusion 1.5 base model."
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, free hair, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin"

for item in tqdm(combined_list, total=len(combined_list)):
# for item in tqdm(conditioning_scale):
    # n = n + 1
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    conditioning_scale, strength, guidance_scale = item
    strength = 0.11 * strength
    # eta = 0.1 * eta
    controlnet_conditioning_scale = conditioning_scale * 0.1
    print(f"kot_inpaint_{round(controlnet_conditioning_scale, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png")
    # generate image
    images = pipe(
        prompt, 
        negative_prompt=neg_prompt, 
        num_inference_steps=num_inference_steps,
        image=image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images
    filename: str = f"/mnt/d/Data/gen_images/kot_inpaint_{round(controlnet_conditioning_scale, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png"
    images[0].save(filename)
