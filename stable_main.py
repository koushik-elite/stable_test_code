import requests
from PIL import Image
from io import BytesIO
from diffusers.utils import make_image_grid, load_image
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image
from diffusers import AutoencoderKL
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

import cv2
import torch
import numpy as np
from diffusers.utils import load_image

from tqdm import tqdm
import itertools
import math
import os.path
import random

num_images_per_prompt: int = 1
num_inference_steps: int = 35
strengths = list(range(3, 10))
conditioning_scale = list(range(1, 9))
guidance_scales = list(range(2, 17))
eta_list = list(range(0, 11))
size_factor: float = 0.99
combined_list = list(itertools.product(conditioning_scale, strengths, guidance_scales))
generator = torch.Generator(device="cuda").manual_seed(2257817932)

device = "cuda"
# model_id_or_path = "acheong08/f222"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
# model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

# vae = AutoencoderKL.from_pretrained(model_id_or_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# pipe = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(device)
# pipe.enable_model_cpu_offload()

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 


init_image = Image.open("assets/images/athena_merge.jpg").convert("RGB")
width, height = init_image.size
ratio = width / height
new_height = 900
new_width = int(ratio * new_height)
print(new_width, new_height)
init_image = init_image.resize((new_width, new_height))
image = np.array(init_image)

low_threshold = 100
high_threshold = 200

cally_image = cv2.Canny(image, low_threshold, high_threshold)
cally_image = cally_image[:, :, None]
cally_image = np.concatenate([cally_image, cally_image, cally_image], axis=2)
cally_image = Image.fromarray(cally_image)

cally_image.save(f"output/canny_images_resize.png")
init_image.save(f"output/init_images_resize.png")
# prompt = "An astronaut riding a green horse"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
# init_image = load_image(url)

# prompt = "Astronaut in a war zone, guns in hands, detailed, 8k, best quality, high quality"

# pass prompt and image to pipeline
# image = pipeline(prompt, image=init_image, strength=0.5).images[0]
# prompt = "Athena protects young man from eros, Greek goddess, Sexy and kinky, hyper realistic, high quality"
prompt = "nude couple, nude lovers, greek women with greek warrior helmet with a young man, goddess, Sexy and kinky, hyper realistic, high quality"
# prompt = "Pallas Athena, seductive smile, Sexy and kinky, ample cleavage, EasyNegative, extremely detailed, (worst quality:2), (low quality:2), (normal quality:2), photo-realistic, high quality, (extremely detailed eyes face and hands)"
# prompt = "Sexy Pallas Athena standing on a globe, holding a spear in her left hand and her shield in her right hand, Roman Soldier Helmet, Nude, boobs, same pose, extremely detailed, photo-realistic, high quality, (extremely detailed eyes face and hands)"
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, lowres, saree cloth, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin, doll"
# prompt = "hindu goddess parvati standing view from backside show the entire toned Booty and cracks, Female Booty Shower, hip, booty, full body shot, best quality, high quality"
# all_images = pipe(prompt=prompt, strength=0.75, guidance_scale=12).images
# all_images = pipe(prompt=prompt, negative_prompt=neg_prompt, image=init_image, num_inference_steps=50, strength=0.8, guidance_scale=12).images

# print(all_images)
n = 0
# for image in all_images:
#     n = n + 1
#     # print(image)
#     image.save(f"output/images_{n}.png")


n = 0
for item in tqdm(combined_list, total=len(combined_list)):
    n = n + 1
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    eta, strength, guidance_scale = item
    strength = 0.11 * strength 
    eta = 0.1 * eta
    print(f"kot_inpaint_{round(eta, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png")
    # generate image
    all_images = pipe(
        prompt=prompt, 
        negative_prompt=neg_prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
        guidance_scale=guidance_scale,
        eta=eta,
    ).images[0]
    filename: str = f"/mnt/d/Data/gen_images/kot_inpaint_{round(eta, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png"
    all_images.save(filename)