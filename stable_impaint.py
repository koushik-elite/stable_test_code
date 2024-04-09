import cv2
from PIL import Image
from diffusers import DDIMScheduler, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from tqdm import tqdm

import itertools
import math
import os.path
import random

from set_seed import seed_everything

# seed: int = 88888
seed: int = 2257817932
seed_everything(seed)
device: str = "cpu"
num_images_per_prompt: int = 1
num_inference_steps: int = 30
strengths = list(range(5, 10))
guidance_scales = list(range(5, 17))
eta_list = list(range(0, 11))
size_factor: float = 0.99
combined_list = list(itertools.product(eta_list, strengths, guidance_scales))
# Shuffle the combined list
random.shuffle(combined_list)

device = "cuda"
# model_id_or_path = "acheong08/f222"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
# model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
# model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
# generator = torch.Generator(device="cpu").manual_seed(2257817932)
# generator = torch.Generator(device="cpu")

# load control net and stable diffusion v1-5
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker = None, torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker = None, torch_dtype=torch.float16)
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype=torch.float16)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.load_lora_weights("/mnt/d/models/clarity_3.safetensors", weight_name="model.safetensors")
pipe.to(device)
# pipe.enable_model_cpu_offload()

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

# get canny image
init_image = Image.open("assets/bust/athena_4.jpg").convert("RGB")
mask_image = Image.open("assets/bust/athena_4_mask.jpg").convert("RGB")
# control_image = make_inpaint_condition(init_image, mask_image)
# control_image = make_canny_condition(init_image)
width, height = init_image.size
# width, height = init_image.size
# ratio = width / height
# new_height = 800
# new_width = int(ratio * new_height)
# print(new_width, new_height)
# init_image = init_image.resize((new_width, new_height))
# init_image.save(f"output/init_images_resize.png")
# init_image = np.array(init_image)

prompt = "Athena standing on a globe, holding a spear in her left hand and her shield in her right hand, Roman Soldier Helmet"
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, free hair, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin"


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
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        eta=eta,
        width=int(math.floor(width * size_factor / 8) * 8),
        height=int(math.floor(height * size_factor / 8) * 8)
    ).images[0]
    filename: str = f"output/gen/kot_inpaint_{round(eta, 4)}_{round(strength, 4)}_{round(guidance_scale, 4)}.png"
    all_images.save(filename)

# n = 0
# for image in all_images:
#     n = n + 1
#     image.save(f"output/images_{n}.png")
