import os
import gc
import torch
from models.StableDiffusion3D.unet import UNet3DConditionModel
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
)


os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


cache_dir = os.getcwd()
my_dtype = torch.float16
model_id = "sd-legacy/stable-diffusion-v1-5"
path = ""


def clear() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_video_model() -> tuple:
    """Zeroscope VDM : returns the pipeline and its frozen sub modules."""
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
         cache_dir = cache_dir,
         torch_dtype = my_dtype
    )
    pipe = pipe.to("cuda")
    video_unet = pipe.unet
    video_text_tokenizer = pipe.tokenizer
    video_text_encoder = pipe.text_encoder
    video_vae = pipe.vae
    video_unet.requires_grad_(False)
    video_text_encoder.requires_grad_(False)
    video_vae.requires_grad_(False)
    return pipe, video_unet, video_text_tokenizer, video_text_encoder, video_vae


def load_image_model() -> tuple:
    """Stable Diffusion 1.5 IDM inflated to 3D : returns the pipeline and its frozen sub modules."""
    pipe_img = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=my_dtype, cache_dir = cache_dir)
    pipe_img = pipe_img.to("cuda")

    tokenizer = pipe_img.tokenizer
    text_encoder = pipe_img.text_encoder
    vae = pipe_img.vae
    vae = vae.requires_grad_(False)
    text_encoder = text_encoder.requires_grad_(False)
    unet = UNet3DConditionModel.from_pretrained_2d(path, torch_dtype = my_dtype)
    unet = unet.requires_grad_(False)
    unet = unet.to("cuda", my_dtype)
    return pipe_img, tokenizer, text_encoder, vae, unet


def edit_frames(frames: torch.Tensor, edit_model_name: str, prompt: str) -> torch.Tensor:
    """Frame wise editing with the image model, frames are in [-1, 1]."""
    if edit_model_name == "InstructPix2Pix" :
        edit_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype = my_dtype,
            safety_checker = None,
        )
        edit_model.scheduler = DPMSolverMultistepScheduler.from_config(edit_model.scheduler.config)
        edit_model.to("cuda")
        frames = edit_model(
            prompt = [prompt] * frames.size(0),
            image = (frames + 1)/2,
            num_inference_steps = 10,
            image_guidance_scale = 5,
            output_type = "pt",
        )[0]
        frames = 2 * (frames - 0.5)

    return frames
