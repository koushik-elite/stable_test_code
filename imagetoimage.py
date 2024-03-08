import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
# model_id_or_path = "acheong08/f222"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = ""

# response = requests.get(url)

init_image = Image.open("assets/myface/8616072.png")
# image.resize((256, 256))
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "best quality, high quality"

all_images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

n = 0
for image in all_images:
    n = n + 1
    image.save(f"output/images_{n}.png")