import cv2
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionControlNetInpaintPipeline
import torch
import numpy as np
from diffusers.utils import load_image
from safetensors.torch import load_file
from transformers import pipeline

from tqdm import tqdm
import itertools
import math
import os.path
import random

num_images_per_prompt: int = 1
num_inference_steps: int = 20
strengths = list(range(5, 10))
conditioning_scale = list(range(1, 9))
guidance_scales = list(range(5, 17))
eta_list = list(range(0, 11))
size_factor: float = 0.99
combined_list = list(itertools.product(conditioning_scale, strengths, guidance_scales))

depth_estimator = pipeline('depth-estimation')

device = "cuda"
# model_id_or_path = "acheong08/f222"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
# model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
generator = torch.Generator(device="cpu").manual_seed(1491891153)


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

init_image = Image.open("assets/bust/roman_sculpture_4.jpg").convert("RGB")
mask_image = Image.open("assets/bust/roman_sculpture_4_mask.jpg").convert("RGB")

depth_image = depth_estimator(init_image)['depth']
depth_image = np.array(depth_image)
depth_image = depth_image[:, :, None]
depth_image = np.concatenate(3*[depth_image], axis=2)
depth_image = Image.fromarray(depth_image)

# control_image = make_inpaint_condition(init_image, mask_image)
# control_image = make_canny_condition(init_image)

controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.to(device)

# width, height = init_image.size
# ratio = width / height
# new_height = 800
# new_width = int(ratio * new_height)
# print(new_width, new_height)
# init_image = init_image.resize((new_width, new_height))
# init_image.save(f"output/init_images_resize.png")
# init_image = np.array(init_image)

# low_threshold = 100
# high_threshold = 200

# canny_image = cv2.Canny(init_image, low_threshold, high_threshold)
# canny_image = canny_image[:, :, None]
# canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
# canny_image = Image.fromarray(canny_image)
# canny_image.save(f"output/init_images_canny.png")

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

prompt = "marble sculpture bust, photo realistic, Ultra realistic, high quality, extremely detailed."
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, free hair, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin"

n = 0
for item in tqdm(combined_list, total=len(combined_list)):
    n = n + 1
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    conditioning_scale, strength, guidance_scale = item
    strength = 0.11 * strength 
    # eta = 0.1 * eta
    eta = 1.0
    conditioning_scale = 0.1 * conditioning_scale
    print(f"kot_inpaint_{round(conditioning_scale, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png")
    # generate image
    all_images = pipe(
        prompt=prompt, 
        negative_prompt=neg_prompt,
        image=init_image,
        controlnet_conditioning_scale = conditioning_scale,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        eta=eta,
        control_image=depth_image,
    ).images[0]
    filename: str = f"output/gen/kot_inpaint_{round(conditioning_scale, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png"
    all_images.save(filename)
