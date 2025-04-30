# Inversion Benchmark

Is GNRI actually better than fixed point, DDIM and DPM inversion for this pipeline? This is the
measurement, run end to end on a Colab T4 with the repo's own `inversion.py` code paths.

## Setup

| | |
|---|---|
| hardware | Tesla T4 16 GB, via [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) `colab new --gpu T4` |
| software | torch 2.11 + cu128, diffusers 0.40, fp16 throughout |
| video branch | `cerspense/zeroscope_v2_576w` UNet3D (the VDM the pipeline samples from) |
| image branch | `stable-diffusion-v1-5` inflated to 3D by the repo's `UNet3DConditionModel.from_pretrained_2d` (the IDM branch) |
| data | 4 clips from [Tune-A-Video](https://github.com/showlab/Tune-A-Video/tree/main/data) - `car-turn`, `man-skiing`, `man-surfing`, `rabbit-watermelon` |
| input | 10 frames, stride 4, 256x256 (the repo's `--num-frames 10 --frame-rate 4 --resolution 256` defaults) |
| prompt | one caption per clip, no classifier free guidance (`do_classifier_free_guidance = False`, as in `main.py`) |

## Protocol

For every method the same latent `z_0` is inverted to `z_T` and then sampled straight back with the
matching forward scheduler and the same prompt - no editing, no latent mixing. A perfect inversion
returns `z_0` exactly, so the gap is inversion error and nothing else.

- **PSNR / LPIPS** are measured against `decode(encode(video))`, the VAE round trip, not against the
  raw frames. That removes the VAE's own loss (~26-32 dB depending on clip) from the number, so the
  metric isolates inversion quality. Higher PSNR / lower LPIPS is better.
- **latent MSE** is `||z_rec - z_0||²` in latent space, the same quantity without the decoder.
- **residual** is `mean |f(z) - z|`, how well the implicit inversion equation is actually solved,
  probed on every 10th timestep. Only comparable *within* the DDIM family (DDIM / FP / GNRI) - DPM
  solves a different `f`, so its residual is not on the same scale.
- **NFE** is unet forward passes, `steps x iters`. The probe calls are excluded from both NFE and the
  timer. This is the axis that matters: FP and GNRI buy accuracy with extra unet calls, so they are
  compared against DDIM/DPM at *equal NFE*, not equal steps.

## Reproducing

```bash
colab new -s inv-bench --gpu T4
colab install -s inv-bench lpips positional_encodings imageio-ffmpeg av
colab upload -s inv-bench repo.tgz /content/repo.tgz
colab exec -s inv-bench --timeout 7200 -f grid_video.py    # scripts in scratchpad/
colab download -s inv-bench /content/results_video.json .
colab stop -s inv-bench
```

