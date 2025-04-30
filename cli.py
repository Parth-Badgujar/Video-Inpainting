import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-vid", type = str)
    parser.add_argument("--output-vid", type = str)
    parser.add_argument("--prompt-vdm", type = str)
    parser.add_argument("--num-frames", type = int, default = 10)
    parser.add_argument("--frame-rate", type = int, default = 4)
    parser.add_argument("--num-steps", type = int, default = 50 )
    parser.add_argument("--scheduler", type = str, default = "DDIM")
    parser.add_argument("--fp-iters", type = int, default = 4)
    parser.add_argument("--gnri-alpha", type = float, default = 0.1)
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

    return args
