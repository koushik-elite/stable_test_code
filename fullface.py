import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter import IPAdapter, IPAdapterFull, IPAdapterPlus

# base_model_path = "runwayml/stable-diffusion-v1-5"
base_model_path = "SG161222/Realistic_Vision_V2.0"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "all_models/IP-Adapter/models/image_encoder/"
ip_ckpt = "all_models/IP-Adapter/models/ip-adapter-full-face_sd15.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

image = Image.open("assets/myface/akk2.png")
# image.resize((256, 256))

# load ip-adapter
# ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
ip_model = IPAdapterFull(pipe, image_encoder_path, ip_ckpt, device, num_tokens=257)
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# all_images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42)

all_images = ip_model.generate(
    pil_image=image, num_samples=5, 
    prompt="A beautiful women, brassiere clothing, best quality, portrait, hyperrealistic",
    num_inference_steps=50, seed=420)

# all_images = ip_model.generate(pil_image=image, num_samples=10, num_inference_steps=100, seed=420,
#         prompt="best quality, A photo of a indian women wearing a bra, upper body")

n = 0
for image in all_images:
    n = n + 1
    image.save(f"output/images_{n}.png")