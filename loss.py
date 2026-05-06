import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights

try:
    from segment_anything import sam_model_registry
except ImportError: 
    sam_model_registry = None


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        vgg_layer_indices=None,
        sam_checkpoint="./weights/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    ):
        super().__init__()

        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layer_indices = [35] if vgg_layer_indices is None else list(vgg_layer_indices)
        max_idx = max(self.vgg_layer_indices)
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(max_idx + 1)])
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        self.sam_encoder = None
        if sam_checkpoint is not None and sam_model_registry is not None:
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.sam_encoder = sam.image_encoder
            for p in self.sam_encoder.parameters():
                p.requires_grad = False

    def forward(self, input, target):

        vgg_loss = input.new_zeros(())
        cur_in, cur_tg = input, target
        for i, layer in enumerate(self.vgg_layers):
            cur_in = layer(cur_in)
            cur_tg = layer(cur_tg)
            if i in self.vgg_layer_indices:
                vgg_loss = vgg_loss + F.mse_loss(cur_in, cur_tg)

        seg_loss = input.new_zeros(())
        if self.sam_encoder is not None:
            sam_in = F.interpolate(input, size=(1024, 1024),
                                   mode="bilinear", align_corners=False)
            sam_tg = F.interpolate(target, size=(1024, 1024),
                                   mode="bilinear", align_corners=False)
            sam_in = self.sam_encoder(sam_in)
            sam_tg = self.sam_encoder(sam_tg)
            seg_loss = F.mse_loss(sam_in, sam_tg)

        return vgg_loss + seg_loss


class VisualLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        # FFT on the spatial dims of each (B, C, H, W) tensor.
        in_fft = torch.fft.fftshift(torch.fft.fft2(input))
        tg_fft = torch.fft.fftshift(torch.fft.fft2(target))

        in_mag = torch.log1p(torch.abs(in_fft))
        tg_mag = torch.log1p(torch.abs(tg_fft))

        # Normalize each spectrum so the loss is scale-invariant.
        in_mag = (in_mag - in_mag.mean()) / (in_mag.std() + self.eps)
        tg_mag = (tg_mag - tg_mag.mean()) / (tg_mag.std() + self.eps)

        return F.mse_loss(in_mag, tg_mag)