import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import DifIISRSampler

from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-input", type=str, default="", help="Input path for inference.")
    parser.add_argument("-output", type=str, default="./results", help="Output path for saving results.")
    parser.add_argument("-reference", type=str, default=None, help="Reference image path")
    parser.add_argument("--config", type=str, default=None, help="Config path.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256],
            help="Chopping forward.",
            )
    
    args = parser.parse_args()
    return args

def get_configs(args):
    configs = OmegaConf.load(args.config)
    
    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
        
    ckpt_path = ckpt_dir / f'DifIISR.pth'
    print(f"[INFO] Using the checkpoint {ckpt_path}")
        
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.output).exists():
        Path(args.output).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop must be in [512, 256]")

    return configs, chop_stride

def main():
    args = get_parser()
    configs, chop_stride = get_configs(args)

    sampler = DifIISRSampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=True,
            seed=args.seed,
            ddim=True
            )
    
    sampler.inference(args.input, args.output, bs=1, noise_repeat=False, one_step=True)
    import evaluate
    evaluate.evaluate(args.output, args.reference, None)
    
    
if __name__ == '__main__':
    main()
