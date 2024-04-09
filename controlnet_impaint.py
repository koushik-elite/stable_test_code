import cv2
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionControlNetInpaintPipeline
import torch
import numpy as np
from diffusers.utils import load_image
from safetensors.torch import load_file

num_images_per_prompt: int = 1
num_inference_steps: int = 30
strengths = list(range(5, 10))
controlnet_conditioning_scale = list(range(1, 10))
guidance_scales = list(range(5, 17))
eta_list = list(range(0, 11))
size_factor: float = 0.99
combined_list = list(itertools.product(eta_list, strengths, guidance_scales))
# Shuffle the combined list
# random.shuffle(combined_list)

device = "cuda"
generator = torch.Generator(device="cpu").manual_seed(1491891153)

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_id_or_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
# pipe = pipe.to(device)
# pipe.enable_model_cpu_offload()
# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/bird.png")

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

init_image = Image.open("assets/bust/the-odyssey.jpg").convert("RGB")
mask_image = Image.open("assets/bust/the-odyssey-mask.jpg").convert("RGB")

control_image = make_inpaint_condition(init_image, mask_image)
# control_image = make_canny_condition(init_image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)
# controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

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

prompt = "marble bust, white flawless, masculine shoulder and arms, hyper details, extremely detailed, photo-realistic, high quality, all on Stable Diffusion 1.5 base model."
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, free hair, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin"

all_images = pipe(prompt=prompt, 
    negative_prompt=neg_prompt,  
    num_inference_steps=35,
    controlnet_conditioning_scale = controlnet_conditioning_scale,
    generator=generator,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    # strength=0.7,
    guidance_scale=9
).images

n = 0
for image in all_images:
    n = n + 1
    image.save(f"output/images_{n}.png")
