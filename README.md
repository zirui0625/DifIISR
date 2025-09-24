# DifIISR: Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.2-%237732a8)](https://pytorch.org/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=zirui0625.DifIISR)

### DifIISR: Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution [CVPR 2025]

<div align=center>
<img src="https://github.com/zirui0625/DifIISR/blob/main/figures/network.png" width="90%">
</div>

## Updates
[2025-3-4] You can find our paper [here](https://arxiv.org/abs/2503.01187).  
[2025-2-27] Our paper has been accepted by CVPR 2025, and the code will be released soon.

## Environment
```
# create virtual environment
conda create -n DifIISR python=3.10
conda activate DifIISR
# install requirements
pip install -r requirements.txt
```

## Test
Our checkpoints can be found in [Google drive](https://drive.google.com/file/d/1PhRvk1Dlp3CCrPkrxRfNbNZ3fNgviDVZ/view?usp=drive_link), put it in 'DifIISR/weights/', you can test our method through
```
CUDA_VISIBLE_DEVICES=0 python inference.py -input dataset/test/LR -output results -reference dataset/test/HR --config configs/DifIISR_test.yaml
```

## Citation
```
@article{li2025difiisr,
  title={DifIISR: A Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution},
  author={Li, Xingyuan and Wang, Zirui and Zou, Yang and Chen, Zhixin and Ma, Jun and Jiang, Zhiying and Ma, Long and Liu, Jinyuan},
  journal={arXiv preprint arXiv:2503.01187},
  year={2025}
}
```
## Contact
If you have any questions, feel free to contact me through <code style="background-color: #f0f0f0;">ziruiwang0625@gmail.com</code>。
## Acknowledgement
Our codes are based on [ResShift](https://github.com/zsyOAOA/ResShift), [Sinsr](https://github.com/CompVis/latent-diffusion), thanks for their contribution.



