import os, sys, math, random
import cv2
import numpy as np
from pathlib import Path

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F

from datapipe.datasets import BaseDataFolder
from utils.util_image import ImageSpliterTh

class BaseSampler:
    def __init__(
            self, configs, sf=4, use_fp16=False, 
            chop_size=256, chop_stride=224, chop_bs=1, 
            desired_min_size=64, seed=3407, ddim=True
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        self.ddim=ddim
        

        self.setup_dist()  

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus== 1, 'Please assign only one available GPU using CUDA_VISIBLE_DEVICES during sampling!'
        self.rank = 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None, 'Please specify the checkpoint path for the model!'
        # load model
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)

        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()       
        # freeze diffusion model
        for params in model.parameters():
            params.requires_grad = False
        
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None, 'Please specify the checkpoint path for the model!'
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                util_net.reload_model(autoencoder, ckpt['state_dict'])
            else:
                util_net.reload_model(autoencoder, ckpt)                
            autoencoder.eval()
            for params in autoencoder.parameters():
                params.requires_grad = False
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

class DifIISRSampler(BaseSampler):    
    def sample_func(self, y0, noise_repeat=False, one_step=False, apply_decoder=True):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB 
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()
        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False


        model_kwargs={'lq':y0,} if self.configs.model.params.cond_lq else None
        #### y0 [1, 3, 512, 512]/[1, 3, 256, 256]
        if not self.ddim:        
            results = self.base_diffusion.p_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=noise_repeat,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        else:
            results = self.base_diffusion.ddim_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=True,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        if flag_pad and apply_decoder:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            
        if not apply_decoder:
            return results["pred_xstart"]
        return results.clamp_(-1.0, 1.0)

    
    def inference(self, input, output, bs=1, noise_repeat=False, one_step=False, return_tensor=False, apply_decoder=True):
        '''
        Inference demo. 
        Input:
            input: str, folder or image path for LQ image
            output: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,  
                        )
                for im_lq_pch, index_infos in im_spliter:
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            noise_repeat=noise_repeat, 
                            one_step=one_step, 
                            apply_decoder=apply_decoder
                            )     
                    im_spliter.update(im_sr_pch.detach(), index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        noise_repeat=noise_repeat, 
                        one_step=one_step, 
                        apply_decoder=apply_decoder
                        )     

            if apply_decoder:
                im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        input = Path(input) if not isinstance(input, Path) else input
        output = Path(output) if not isinstance(output, Path) else output
        
        if self.rank == 0:
            assert input.exists(),'Input path does not exist'
            if not output.exists():
                output.mkdir(parents=True)
        
        return_res = {}
        if input.is_dir():
            data_config =   {'params': {'dir_path': str(input),
                                    'transform_type': 'default',
                                    'transform_kwargs': {
                                        'mean': 0.0,
                                        'std': 1.0,
                                        },
                                    'need_path': True,
                                    'recursive': True,
                                    'length': None,
                                    }
                            }
            dataset = BaseDataFolder(**data_config['params'])
            self.write_log(f'Find {len(dataset)} images in {input}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for data in dataloader:
                results = _process_per_image(data['lq'].cuda())    
                for jj in range(results.shape[0]):
                    im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                    im_name = Path(data['path'][jj]).stem
                    im_path = output / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
                    if return_tensor:
                        return_res[im_path.stem]=results[jj]
                        
        else:
            im_lq = util_image.imread(input, chn='rgb', dtype='float32')  
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              
            im_sr_tensor = _process_per_image(im_lq_tensor)
            im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

            im_path = output / f"{input.stem}.png"
            util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
            if return_tensor:
                return_res[im_path.stem]=im_sr_tensor

        self.write_log(f"Processing done, enjoy the results in {str(output)}")
        return return_res
    
if __name__ == '__main__':
    pass
