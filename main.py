#!/usr/bin/env python
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--input-vid", type = str)
parser.add_argument("--output-vid", type = str)
parser.add_argument("--prompt-vdm", type = str)
parser.add_argument("--num-frames", type = int, default = 10)
parser.add_argument("--frame-rate", type = int, default = 4)
parser.add_argument("--num-steps", type = int, default = 50 )
parser.add_argument("--scheduler", type = str, default = "DDIM")
parser.add_argument("--fp-iters", type = int, default = 4)
parser.add_argument("--scale-model-inp", action='store_true')
parser.add_argument("--mixing-ratio", type = float, default = 0.1)
parser.add_argument("--prompt-idm", type = str)
parser.add_argument("--resolution", type = int, default = 256)
parser.add_argument("--guidance-scale", type = float, default = 5.0)
parser.add_argument("--edit-model", type = str, default = "InstructPix2Pix")
parser.add_argument("--mask-path", type = str, default = None)





args = parser.parse_args()

if "inpainting" in args.edit_model : 
    assert args.mask_path != None, "Cannot perform inpainting without mask"

import os
import gc
import torch
import imageio
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from models.util import read_video
from diffusers.utils import export_to_video
from models.video_smoothing import VideoSmoothingPipeline
from models.StableDiffusion3D.unet import UNet3DConditionModel
from diffusers import AutoencoderKL as AutoencoderKL_zeroscope
from models.lib.diffusers_v23.models.unet_3d_condition import UNet3DConditionModel as UNet3DConditionModel_zeroscope
from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline, 
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline, 
    DDIMScheduler, 
    DDIMInverseScheduler, 
    DPMSolverMultistepInverseScheduler, 
    DPMSolverMultistepScheduler, 
) 


def clear():
    gc.collect()
    torch.cuda.empty_cache()


os.environ["HF_TOKEN"] = "" 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


cache_dir = os.getcwd()
my_dtype = torch.float16


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



path = "/teamspace/studios/this_studio/BIVDiff/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/"


model_id = "sd-legacy/stable-diffusion-v1-5"
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



clear()



video_path = args.input_vid
mask_path = args.mask_path
width = args.resolution
height = args.resolution
frame_rate = args.frame_rate
num_inference_steps = args.num_steps
video_length = args.num_frames


frames = read_video(
    video_path, 
    video_length = video_length, 
    width = width, 
    height = height, 
    frame_rate = frame_rate
)
original_pixels = rearrange(frames, "(b f) c h w -> b c f h w", b=1)
frames = frames.to("cuda", my_dtype)  


if args.edit_model == "InstructPix2Pix" : 
    edit_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", 
        torch_dtype = my_dtype, 
        safety_checker = None, 
    )
    edit_model.scheduler = DPMSolverMultistepScheduler.from_config(edit_model.scheduler.config) 
    edit_model.to("cuda")
    prompt = args.prompt_vdm  
    frames = edit_model(
        prompt = [prompt] * frames.size(0), 
        image = (frames + 1)/2, 
        num_inference_steps = 10, 
        image_guidance_scale = 5, 
        output_type = "pt", 
    )[0] 
    frames = 2 * (frames - 0.5) 


latents = vae.encode(frames).latent_dist.sample() * 0.18215 


prompt = args.prompt_idm
device = "cuda"
do_classifier_free_guidance = False 


prompt_embeds_video = pipe.encode_prompt(
    prompt,
    device,
    1,
    do_classifier_free_guidance,
)

prompt_embeds_img = pipe_img.encode_prompt(
    prompt,
    device,
    1,
    do_classifier_free_guidance,
)

if args.scheduler in ["DDIM", "FP"]:
    scheduler1 = DDIMInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
    scheduler1.set_timesteps(num_inference_steps, device=device)
    scheduler2 = DDIMInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
    scheduler2.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler1.timesteps

elif args.scheduler in ["DPM"] : 
    scheduler1 = DPMSolverMultistepInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
    scheduler1.set_timesteps(num_inference_steps, device=device)
    scheduler2 = DPMSolverMultistepInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
    scheduler2.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler1.timesteps  


clear()


def gnri_optimize(latent, t, prompt_embeds, unet, num_iter, scheduler, z_0, do_classifier_free_guidance=False, guidance_scale=1.0, alpha = 0.2):
    last = latent
    latent_0 = latent.clone()
    best_latent = None
    alpha_cp = scheduler.alphas_cumprod[t] 
    best_score = torch.inf
    for i in range(num_iter):
        latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
        if scale_model_inp :
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
            latent = scheduler.step(noise_pred, t, latent_0).prev_sample

        latent = latent.requires_grad_(True)
        fx = (latent - last).abs()# + alpha * (torch.pow(latent - alpha_cp * z_0, 2) / (1 - alpha_cp))
        l = fx.sum() 
        score = fx.mean() 
        print(score)
        if score < best_score:
            best_score = score
            best_latent = latent.detach()
        
        l.backward()
        latent = latent - (1 / (64 * 64 * 4)) * (l / (latent.grad ))
        latent.grad = None
        latent._grad_fn = None
        last = latent 
        

    return latent




def optimize(latent, t, prompt_embeds, unet, num_iter, scheduler, do_classifier_free_guidance=False, guidance_scale=1.0):
        last = latent
        latent_0 = latent.clone()
        for i in range(num_iter):
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            if scale_model_inp :
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t, latent_0).prev_sample

            score = torch.norm(last - latent)
            last = latent

        return latent


fixed_point = False
if args.scheduler == "FP":
    fixed_point = True
scale_model_inp = args.scale_model_inp
num_fp_iters = args.fp_iters



latents = latents[None, ...].permute(0, 2, 1, 3, 4)
latents_img = latents.clone()
latents_video = latents.clone()

print("Started Mixed inversion")
for iteration, t in  tqdm(enumerate(timesteps), total = num_inference_steps):
    if fixed_point : 
        latents_img = optimize(latents_img, t, prompt_embeds_img[0], unet, num_iter = num_fp_iters, scheduler = scheduler1)
        latents_video = optimize(latents_video, t, prompt_embeds_video[0], unet = video_unet, num_iter = num_fp_iters, scheduler = scheduler2)

        # latents_img = gnri_optimize(
        #     latents_img, 
        #     t, 
        #     prompt_embeds_img[0], 
        #     unet, 
        #     num_iter = num_fp_iters, 
        #     scheduler = scheduler1, 
        #     z_0 = latents.clone(), 
        #     alpha = 0.02
        # )

        # latents_video = gnri_optimize(
        #     latents_video, 
        #     t, 
        #     prompt_embeds_video[0], 
        #     video_unet, 
        #     num_iter = num_fp_iters, 
        #     scheduler = scheduler2, 
        #     z_0 = latents.clone(), 
        #     alpha = 0.02
        # )

    else : 
        latents_img_input = scheduler1.scale_model_input(latents_img, t) if scale_model_inp else latents_img
        latents_img = scheduler1.step(
            unet(latents_img_input, t, prompt_embeds_img[0]).sample, t,
            latents_img, return_dict = False)[0]

        latents_video_input = scheduler2.scale_model_input(latents_video, t) if scale_model_inp else latents_video
        latents_video_input = scheduler2.scale_model_input(latents_video, t)        
        latents_video = scheduler2.step(
            video_unet(latents_video_input, t, prompt_embeds_video[0]).sample, t,
            latents_video, return_dict = False)[0]
        
clear()


print("Started sampling")

if args.scheduler in ["FP", "DDIM"] :
    normal_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder = "scheduler")
elif args.scheduler == "DPM" : 
    normal_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder = "scheduler")

normal_scheduler.set_timesteps(args.num_steps)
video_smoothing_pipeline = VideoSmoothingPipeline(vae=vae, text_encoder=video_text_encoder,
                      tokenizer=video_text_tokenizer, unet=video_unet,
                      scheduler=normal_scheduler,)


mixing_ratio = args.mixing_ratio
latents = mixing_ratio * latents_img + (1 - mixing_ratio) * latents_video 

video = video_smoothing_pipeline(
    prompt=args.prompt_vdm,
    video_length=args.num_frames,
    num_inference_steps=args.num_steps,
    guidance_scale=args.guidance_scale,
    width=args.resolution, height=args.resolution,
    latents=latents,
).videos

video = video[0].permute(1, 2, 3, 0)
video = video.cpu().numpy()
path = export_to_video(video, output_video_path = args.output_vid, fps = 4)

print("Video saved at :", path)



