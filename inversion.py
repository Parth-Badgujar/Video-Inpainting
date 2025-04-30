import torch
from tqdm.auto import tqdm
from pipelines import model_id
from diffusers import (
    DDIMScheduler,
    DDIMInverseScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
)


def build_inverse_schedulers(scheduler_name: str, num_inference_steps: int, device: str) -> tuple:
    """One inverse scheduler per branch, image and video, sharing the same timesteps."""
    if scheduler_name in ["DDIM", "FP", "GNRI"]:
        scheduler1 = DDIMInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
        scheduler1.set_timesteps(num_inference_steps, device=device)
        scheduler2 = DDIMInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
        scheduler2.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler1.timesteps

    elif scheduler_name in ["DPM"] :
        scheduler1 = DPMSolverMultistepInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
        scheduler1.set_timesteps(num_inference_steps, device=device)
        scheduler2 = DPMSolverMultistepInverseScheduler.from_pretrained(model_id, subfolder = "scheduler")
        scheduler2.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler1.timesteps

    return scheduler1, scheduler2, timesteps


def build_sampling_scheduler(scheduler_name: str, num_inference_steps: int):
    if scheduler_name in ["FP", "GNRI", "DDIM"] :
        normal_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder = "scheduler")
    elif scheduler_name == "DPM" :
        normal_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder = "scheduler")

    normal_scheduler.set_timesteps(num_inference_steps)
    return normal_scheduler


def gnri_optimize(latent, t, prompt_embeds, unet, num_iter, scheduler, z_0, scale_model_inp=False, do_classifier_free_guidance=False, guidance_scale=1.0, alpha = 0.1):
    """Guided Newton Raphson Inversion, ICLR 2025.

    Solves the implicit DDIM inversion step z = f(z) as the root of the scalar
    F(z) = ||f(z) - z||_1 - alpha * log q(f(z) | z_0), where the guidance term
    q(. | z_0) = N(sqrt(alpha_cp) * z_0, 1 - alpha_cp) keeps the root inside the
    distribution the model was trained on. The Jacobian collapses to the sign of
    the residual, so no gradient ever flows through the unet.

    alpha_cp is taken at the timestep of the solved latent, the next one on the
    inverse trajectory, otherwise the guidance is scored against a variance that
    is an order of magnitude too small at the low timesteps.
    """
    latent_0 = latent.clone()
    best_latent = None
    best_score = torch.inf
    next_index = min(int((scheduler.timesteps == t).nonzero()) + 1, len(scheduler.timesteps) - 1)
    alpha_cp = scheduler.alphas_cumprod.to(latent.device)[scheduler.timesteps[next_index]]
    mean = torch.sqrt(alpha_cp) * z_0.float()
    sigma = torch.sqrt(1 - alpha_cp)
    for i in range(num_iter):
        latent = latent.detach().requires_grad_(True)
        with torch.no_grad():
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

            next_latent = scheduler.step(noise_pred, t, latent_0).prev_sample

        log_q = - 0.5 * torch.pow((next_latent.float() - mean) / sigma, 2)
        fx = (next_latent - latent).abs().float() - alpha * log_q
        l = fx.sum()
        score = fx.mean()

        if score < best_score:
            best_score = score
            best_latent = next_latent.detach()

        l.backward()
        # the residual is an l1 norm so the derivative is the sign of it, a coordinate
        # already sitting on the root has a zero derivative and simply does not move
        newton_step = torch.where(latent.grad != 0, l / latent.grad, torch.zeros_like(l))
        latent = (latent - (1 / latent.numel()) * newton_step).to(latent.dtype)

    return best_latent


def optimize(latent, t, prompt_embeds, unet, num_iter, scheduler, scale_model_inp=False, do_classifier_free_guidance=False, guidance_scale=1.0):
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


def mixed_inversion(latents, timesteps, prompt_embeds_img, prompt_embeds_video, unet, video_unet,
                    scheduler1, scheduler2, mode, scale_model_inp, num_fp_iters, gnri_alpha = 0.1) -> tuple:
    """Invert the same latents twice, once through the image unet and once through the video unet."""
    z_0 = latents.clone()
    latents_img = latents.clone()
    latents_video = latents.clone()

    for iteration, t in  tqdm(enumerate(timesteps), total = len(timesteps)):
        if mode == "FP" : 
            latents_img = optimize(latents_img, t, prompt_embeds_img[0], unet, num_iter = num_fp_iters, scheduler = scheduler1, scale_model_inp = scale_model_inp)
            latents_video = optimize(latents_video, t, prompt_embeds_video[0], unet = video_unet, num_iter = num_fp_iters, scheduler = scheduler2, scale_model_inp = scale_model_inp)

        elif mode == "GNRI" : 
            latents_img = gnri_optimize(
                latents_img, 
                t, 
                prompt_embeds_img[0], 
                unet, 
                num_iter = num_fp_iters, 
                scheduler = scheduler1, 
                z_0 = z_0, 
                scale_model_inp = scale_model_inp, 
                alpha = gnri_alpha
            )

            latents_video = gnri_optimize(
                latents_video, 
                t, 
                prompt_embeds_video[0], 
                video_unet, 
                num_iter = num_fp_iters, 
                scheduler = scheduler2, 
                z_0 = z_0, 
                scale_model_inp = scale_model_inp, 
                alpha = gnri_alpha
            )

        else : 
            latents_img_input = scheduler1.scale_model_input(latents_img, t) if scale_model_inp else latents_img
            latents_img = scheduler1.step(
                unet(latents_img_input, t, prompt_embeds_img[0]).sample, t,
                latents_img, return_dict = False)[0]

            latents_video_input = scheduler2.scale_model_input(latents_video, t) if scale_model_inp else latents_video
            latents_video = scheduler2.step(
                video_unet(latents_video_input, t, prompt_embeds_video[0]).sample, t,
                latents_video, return_dict = False)[0]

    return latents_img, latents_video
