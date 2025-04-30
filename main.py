#!/usr/bin/env python
from cli import parse_args

args = parse_args()

from einops import rearrange
from models.util import read_video
from diffusers.utils import export_to_video
from models.video_smoothing import VideoSmoothingPipeline
from pipelines import clear, load_video_model, load_image_model, edit_frames, my_dtype
from inversion import build_inverse_schedulers, build_sampling_scheduler, mixed_inversion


pipe, video_unet, video_text_tokenizer, video_text_encoder, video_vae = load_video_model()
pipe_img, tokenizer, text_encoder, vae, unet = load_image_model()

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


frames = edit_frames(frames, args.edit_model, args.prompt_vdm)


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

scheduler1, scheduler2, timesteps = build_inverse_schedulers(args.scheduler, num_inference_steps, device)

clear()


mode = args.scheduler
scale_model_inp = args.scale_model_inp
num_fp_iters = args.fp_iters


latents = latents[None, ...].permute(0, 2, 1, 3, 4)

print("Started Mixed inversion")
latents_img, latents_video = mixed_inversion(
    latents, 
    timesteps, 
    prompt_embeds_img, 
    prompt_embeds_video, 
    unet, 
    video_unet, 
    scheduler1, 
    scheduler2, 
    mode = mode, 
    scale_model_inp = scale_model_inp, 
    num_fp_iters = num_fp_iters, 
    gnri_alpha = args.gnri_alpha, 
)

clear()


print("Started sampling")

normal_scheduler = build_sampling_scheduler(args.scheduler, args.num_steps)
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
