import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL
import torch
import numpy as np
from diffusers.utils import load_image

device = "cuda"
# model_id_or_path = "acheong08/f222"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
# model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet_conditioning_scale = 0.71  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_id_or_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
# pipe = pipe.to(device)
pipe.enable_model_cpu_offload()
# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/bird.png")

init_image = Image.open("assets/myface/Atenea-K19-05-b.png").convert("RGB")
width, height = init_image.size
ratio = width / height
new_height = 800
new_width = int(ratio * new_height)
print(new_width, new_height)
init_image = init_image.resize((new_width, new_height))
# init_image.save(f"output/init_images_resize.png")
init_image = np.array(init_image)

low_threshold = 100
high_threshold = 200

canny_image = cv2.Canny(init_image, low_threshold, high_threshold)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)
canny_image.save(f"output/init_images_canny.png")

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

prompt = "a women sculpture with fully exposed shoulder chest and boobs, broad shoulder, midriff, waist, extremely detailed, photo-realistic, high quality, (extremely detailed eyes face and hands)"
# prompt = "Sexy Pallas Athena standing on a globe, holding a spear in her left hand and her shield in her right hand, Roman Soldier Helmet, Nude, boobs, same pose, extremely detailed, photo-realistic, high quality, (extremely detailed eyes face and hands)"
neg_prompt  = "ugly, facial hair, body hair, deformed, disfigured, poor details, bad anatomy, lowres, mutant, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border, porcelain skin, doll"
# prompt = "hindu goddess parvati standing view from backside show the entire toned Booty and cracks, Female Booty Shower, hip, booty, full body shot, best quality, high quality"
# all_images = pipe(prompt=prompt, strength=0.75, guidance_scale=12).images
all_images = pipe(prompt=prompt, negative_prompt=neg_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, num_inference_steps=20, strength=0.8, guidance_scale=12).images

n = 0
for image in all_images:
    n = n + 1
    image.save(f"output/images_{n}.png")
