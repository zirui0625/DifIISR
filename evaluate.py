import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(input, reference, ntest):
    
    model_name="DifIISR    "
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_paired_dict = {}
    
    input = Path(input) if not isinstance(input, Path) else input
    assert input.is_dir()
    
    reference_list = None
    if reference is not None:
        reference = Path(reference) if not isinstance(reference, Path) else reference
        reference_list = sorted([x for x in reference.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: reference_list = reference_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_list = sorted([x for x in input.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_list = lr_list[:ntest]
    
    print(f'Find {len(lr_list)} images in {input}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_list))):
        input = lr_list[i]
        reference = reference_list[i] if reference_list is not None else None
        
        im_in = util_image.imread(input, chn='GREY', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
        
        if reference is not None:
            im_ref = util_image.imread(reference, chn='GREY', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
    print("\n"*2+"="*80)
    print("\t\tCLIPIQA\tMUSIQ \tniqe\tpsnr\tlpips\tssim")
    print(f"{model_name}\t"
      f"{result.get('clipiqa', 0)/len(lr_list):.4f}\t"
      f"{result.get('musiq', 0)/len(lr_list):.3f}\t"
      f"{result.get('niqe', 0)/len(lr_list):.4f}\t"
      f"{result.get('psnr', 0)/len(lr_list):.3f}\t"
      f"{result.get('lpips', 0)/len(lr_list):.4f}\t"
      f"{result.get('ssim', 0)/len(lr_list):.4f}")
    print("="*80)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.input, args.reference, args.ntest)
    


