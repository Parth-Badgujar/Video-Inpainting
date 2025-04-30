# Video-Inpainting

Training free video editing / inpainting built on **BIVDiff** (*Bridging Image and Video Diffusion Models*, CVPR 2024),
with **Fixed Point** and **Guided Newton Raphson** inversion added to the inversion stage.

## Method

1. **Frame wise editing (IDM)** - every frame is edited independently by an image model (InstructPix2Pix by default) using `--prompt-vdm`.
2. **Mixed Inversion** - the edited latents are inverted twice with `--prompt-idm`, once through the inflated SD 1.5 UNet (image branch) and once through the Zeroscope UNet (video branch). The two noises are blended by `--mixing-ratio` (`ratio * image + (1 - ratio) * video`), trading frame fidelity against temporal consistency.
3. **Temporal smoothing (VDM)** - the mixed noise is denoised by the Zeroscope video diffusion model (`VideoSmoothingPipeline`), which removes the flicker left by the per-frame edit.

## Inversion (`--scheduler`)

Each inverse step is really the implicit equation `z_{t+1} = f(z_{t+1})`, which plain DDIM solves with a single approximation. The three options differ in how well they solve it, at `--fp-iters` unet calls per timestep.

| flag | what it does |
|---|---|
| `DDIM` / `DPM` | one inverse step per timestep, no root finding |
| `FP` | **Fixed Point Inversion** - iterate `z <- f(z)` until it stops moving |
| `GNRI` | **Guided Newton Raphson Inversion** ([ICLR 2025](https://arxiv.org/abs/2312.12540)) - Newton root finding on the residual, plus a Gaussian guidance term |

GNRI minimizes `F(z) = ||f(z) - z||_1 - alpha * log q(f(z) | z_0)`, where `q(. | z_0) = N(sqrt(a_t) * z_0, 1 - a_t)` is the forward diffusion posterior. The guidance keeps the root inside the latent distribution the model was trained on, which is what plain Newton drifts away from. Because the residual is an L1 norm its Jacobian collapses to a sign, so the Newton step costs nothing extra - no gradient ever flows through the unet. `--gnri-alpha` is the weight (paper default `0.1`), and 2 iterations are usually enough.

```
--scheduler GNRI --fp-iters 2 --gnri-alpha 0.1
```

## Layout

| file | what it holds |
|---|---|
| `main.py` | the pipeline end to end : load, edit, invert, mix, sample, export |
| `cli.py` | argument parsing |
| `pipelines.py` | model loading (Zeroscope VDM, SD 1.5 IDM, editing model) |
| `inversion.py` | schedulers, fixed point and GNRI root finding, mixed inversion loop |
| `models/` | vendored pipelines : `video_smoothing.py`, `mixed_inversion.py`, `StableDiffusion3D/`, editing backends |

Set `path` in `pipelines.py` to the SD 1.5 checkpoint that `UNet3DConditionModel.from_pretrained_2d` should inflate.

## Usage

```
python main.py --input-vid in.mp4 --output-vid out.mp4 \
               --prompt-idm "a man walking" --prompt-vdm "make him a bronze statue" \
               --scheduler GNRI --fp-iters 2 --num-steps 50 --mixing-ratio 0.1
```

```
options:
  --input-vid INPUT_VID          source video
  --output-vid OUTPUT_VID        destination path
  --prompt-idm PROMPT_IDM        prompt used for inversion (describes the source)
  --prompt-vdm PROMPT_VDM        edit prompt, used by the editing model and the VDM
  --num-frames NUM_FRAMES        frames to sample from the video (default 10)
  --frame-rate FRAME_RATE        sampling stride (default 4)
  --num-steps NUM_STEPS          inversion / sampling steps (default 50)
  --scheduler SCHEDULER          DDIM | FP | GNRI | DPM (default DDIM)
  --fp-iters FP_ITERS            root finding iterations per step, FP / GNRI (default 4)
  --gnri-alpha GNRI_ALPHA        guidance weight for GNRI (default 0.1)
  --scale-model-inp              scale the unet input with the scheduler
  --mixing-ratio MIXING_RATIO    weight of the image branch latents (default 0.1)
  --resolution RESOLUTION        square resolution (default 256)
  --guidance-scale GUIDANCE_SCALE  (default 5.0)
  --edit-model EDIT_MODEL        editing backend (default InstructPix2Pix)
  --mask-path MASK_PATH          required when the edit model is an inpainting one
```
