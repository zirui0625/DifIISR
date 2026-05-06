import enum
import math

import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image

from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from ldm.models.autoencoder import AutoencoderKLTorch

from loss import PerceptualLoss, VisualLoss


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """Get a pre-defined beta schedule for the given name."""
    if schedule_name == "linear":
        return np.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64
        ) ** 2
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """Get a pre-defined eta schedule for the given name."""
    if schedule_name == "exponential":
        power = kwargs.get("power", None)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == "ldm":
        import scipy.io as sio
        mat_path = kwargs.get("mat_path", None)
        sqrt_etas = sio.loadmat(mat_path)["sqrt_etas"].reshape(-1)
    else:
        raise ValueError(f"Unknown schedule_name {schedule_name}")
    return sqrt_etas


class ModelMeanType(enum.Enum):
    """Which type of output the model predicts."""
    START_X = enum.auto()        # the model predicts x_0
    EPSILON = enum.auto()        # the model predicts epsilon
    PREVIOUS_X = enum.auto()     # the model predicts x_{t-1}
    RESIDUAL = enum.auto()       # the model predicts the residual (y - x_0)
    EPSILON_SCALE = enum.auto()  # the model predicts a scaled epsilon


class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL


class ModelVarTypeDDPM(enum.Enum):
    """What is used as the model's output variance."""
    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
        use_guidance=True,
        start_guidance=50,
        warmup_iters=0,
        apply_freq=1,
        use_perceptual=True,
        W_per=0.001,
        use_visual=True,
        W_vis=0.001,
        rho=1.0,
        sam_checkpoint="./weights/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        self.use_guidance = use_guidance
        self.start_guidance = start_guidance
        self.warmup_iters = int(warmup_iters)
        self.apply_freq = max(1, int(apply_freq))
        self.use_perceptual = use_perceptual
        self.W_per = float(W_per)
        self.use_visual = use_visual
        self.W_vis = float(W_vis) 
        self.rho = float(rho)
        self._current_iters = 0
        self.perceptual_loss_fn = PerceptualLoss(
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
        ).cuda()
        self.perceptual_loss_fn.eval()
        self.visual_loss_fn = VisualLoss().cuda()

        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas ** 2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # coefficient for DDIM inference
        self.etas_prev_clipped = np.append(self.etas_prev[1], self.etas_prev[1:])
        self.ddim_coef1 = self.etas_prev * self.etas
        self.ddim_coef2 = self.etas_prev / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                kappa * self.alpha / ((1 - self.etas) * self.sqrt_etas)
            ) ** 2
        else:
            raise NotImplementedError(model_mean_type)
        self.weight_loss_mse = weight_loss_mse

    def set_current_iters(self, iters):
        self._current_iters = int(iters)

    def _guidance_active(self, t):
        if not self.use_guidance:
            return False
        if not (self.use_visual or self.use_perceptual):
            return False
        if self._current_iters <= self.warmup_iters:
            return False
        if self.apply_freq > 1 and (self._current_iters % self.apply_freq != 0):
            return False
        # Only at the heaviest-noise step.
        if not bool((t == (self.num_timesteps - 1)).all()):
            return False
        return True

    def q_mean_variance(self, x_start, y, t):
        """Get the distribution q(x_t | x_0)."""
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa ** 2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        """Diffuse the data for a given number of diffusion steps."""
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute the mean and variance of q(x_{t-1} | x_t, x_0)."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """Apply the model to get p(x_{t-1} | x_t) and a prediction of x_0."""
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # ddim coef
        ddim_coef1 = _extract_into_tensor(self.ddim_coef1, t, x_t.shape)
        ddim_coef2 = _extract_into_tensor(self.ddim_coef2, t, x_t.shape)
        etas = _extract_into_tensor(self.etas, t, x_t.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x_t.shape)
        k = (1 - etas_prev + th.sqrt(ddim_coef1) - th.sqrt(ddim_coef2))
        m = th.sqrt(ddim_coef2)
        j = (etas_prev - th.sqrt(ddim_coef1))

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_noise = process_xstart(
            self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
        )

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )
        else:
            raise ValueError(f"Unknown Mean type: {self.model_mean_type}")

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t,
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_noise": pred_noise,
            # used for ddim
            "ddim_k": k,
            "ddim_m": m,
            "ddim_j": j,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
            - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return y - residual

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
            - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def ddim_inverse(self, model, x, y, t, clip_denoised=True, denoised_fn=None,
                     model_kwargs=None, noise_repeat=False):
        out = self.p_mean_variance(
            model, x, y, t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = (x - pred_xstart * out["ddim_k"] - out["ddim_j"] * y) / out["ddim_m"]
        return {"sample": sample, "pred_xstart": pred_xstart}

    def p_sample(self, model, ground_truth, x, y, t,
                 clip_denoised=True, denoised_fn=None, model_kwargs=None,
                 noise_repeat=False):
        """Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(
            model, x, y, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0, ].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}


    def cond_fn(self, x, y, t_, pred_noise, ground_truth, first_stage_model=None):
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2]
        # print(f"[cond_fn] called from {caller.filename}:{caller.lineno} ({caller.name})")
        mask = None
        with th.enable_grad():
            x_t = x.detach().requires_grad_(True)
            x_start_latent = self._predict_xstart_from_eps(x_t, y, t_, pred_noise)

            # Decode to image space so that VGG / SAM see RGB pixels.
            if first_stage_model is not None:
                x_start_img = self.decode_first_stage(
                    x_start_latent, first_stage_model, no_grad=False,
                )
            else:
                x_start_img = x_start_latent

            loss = x_start_img.new_zeros(())
            if self.use_visual:
                loss = loss + self.W_vis * self.visual_loss_fn(x_start_img, ground_truth)
            if self.use_perceptual:
                loss = loss + self.W_per * self.perceptual_loss_fn(x_start_img, ground_truth)

            gradient = th.autograd.grad(loss.sum(), x_t)[0]
        return gradient, mask

    def ddim_inverse_loop(
        self, x, y, model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.ddim_inverse_loop_progressive(
            x, y, model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
        ):
            final = sample["sample"]
        return final

    def inverse_reflow(
        self, x, y, model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        hyper=1,
    ):
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_x = self.encode_first_stage(x, first_stage_model, up_sample=False)

        t = th.tensor([0] * y.shape[0], device=device)
        out = self.p_mean_variance(
            model, z_x, z_y, t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        return (z_x - pred_xstart) / hyper

    def p_sample_loop(
        self, y, ground_truth, model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        one_step=False,
        apply_decoder=True,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            y, ground_truth, model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            one_step=one_step,
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final["sample"]

    def ddim_inverse_loop_progressive(
            self, x, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_x = self.encode_first_stage(x, first_stage_model, up_sample=False)

        indices = list(range(1, self.num_timesteps))
        z_sample = z_x

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_inverse(
                    model, z_sample, z_y, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                yield out
                z_sample = out["sample"]

    def p_sample_loop_progressive(
            self, y, ground_truth, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            one_step=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0, ].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model, ground_truth, z_sample, z_y, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]

    def decode_first_stage(self, z_sample, first_stage_model=None, no_grad=True):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        if no_grad:
            with th.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
        else:
            z_sample = 1 / self.scale_factor * z_sample
            z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
            out = first_stage_model.decode(z_sample, grad_forward=True)
        return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode="bicubic")
        if first_stage_model is None:
            return y
        with th.no_grad():
            y = y.type(dtype=next(first_stage_model.parameters()).dtype)
            z_y = first_stage_model.encode(y)
            out = z_y * self.scale_factor
        return out.type(ori_dtype)

    def prior_sample(self, y, noise=None):
        """Generate samples from the prior distribution q(x_T|x_0) ~= N(x_T|y, ~)."""
        if noise is None:
            noise = th.randn_like(y)
        t = th.tensor([self.num_timesteps - 1, ] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses_distill(
            self, model, teacher_model, x_start, y, t,
            use_perceptual, W_per, use_visual, W_vis,
            first_stage_model=None, model_kwargs=None,
            noise=None, distill_ddpm=False,
            uncertainty_hyper=False, uncertainty_num_aux=2, learn_xT=False,
            finetune_use_gt=False, xT_cov_loss=False, reformulated_reflow=False,
            loss_in_image_space=False,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        # Allow the caller to override per-call (kept for backward compat).
        if use_perceptual is not None:
            self.use_perceptual = use_perceptual
        if use_visual is not None:
            self.use_visual = use_visual
        if W_per is not None:
            self.W_per = float(W_per)
        if W_vis is not None:
            self.W_vis = float(W_vis)

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        ground_truth = self.encode_first_stage(x_start, first_stage_model, up_sample=False)
        if noise is None:
            noise = th.randn_like(z_y)

        terms = {}
        loss_type = "mse"
        assert loss_type in ["mse", "mae"]
        terms["loss"] = 0
        z_t = self.prior_sample(z_y, noise)
        pred_zstart = None

        if distill_ddpm:
            z_start_teacher = self.p_sample_loop(
                y, x_start, teacher_model, first_stage_model, noise,
                clip_denoised=True if first_stage_model is None else False,
                apply_decoder=False, model_kwargs=model_kwargs,
            )["sample"]
        else:
            z_start_teacher = self.ddim_sample_loop(
                y, x_start, teacher_model, noise, first_stage_model,
                clip_denoised=True if first_stage_model is None else False,
                apply_decoder=False, model_kwargs=model_kwargs,
            )["sample"]

        if self.loss_type not in (LossType.MSE, LossType.WEIGHTED_MSE):
            raise NotImplementedError(self.loss_type)

        model_output = model(self._scale_input(z_t, t), t, **model_kwargs)

        if uncertainty_hyper:
            with th.no_grad():
                model_output_aux_list = []
                for _ in range(uncertainty_num_aux):
                    z_t_aux = self.q_sample(z_start_teacher, z_y, t, noise=th.randn_like(z_y))
                    model_output_aux_list.append(
                        model(self._scale_input(z_t_aux, t), t, **model_kwargs)
                    )
                model_output_aux = th.stack(model_output_aux_list, dim=0)
                uncertainty = (
                    model_output_aux.max(dim=0)[0] - model_output_aux.min(dim=0)[0]
                ).max(dim=1, keepdim=True)[0]
                z_start_gt = self.encode_first_stage(x_start, first_stage_model, up_sample=False)
                uncertainty = (uncertainty * uncertainty_hyper).clip(0, 1)
                z_start = z_start_teacher * uncertainty + z_start_gt * (1 - uncertainty)
        else:
            z_start = z_start_teacher

        if self._guidance_active(t):
            # print(f"[_guidance_active] _current_iters={self._current_iters}, warmup_iters={self.warmup_iters}, use_guidance={self.use_guidance}, use_visual={self.use_visual}, use_perceptual={self.use_perceptual}")
            if self.model_mean_type == ModelMeanType.EPSILON:
                pred_eps = model_output
            elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                scale = _extract_into_tensor(
                    self.kappa * self.sqrt_etas, t, model_output.shape
                )
                pred_eps = model_output / scale
            elif self.model_mean_type == ModelMeanType.START_X:
                pred_eps = self._predict_eps_from_xstart(z_t, z_y, t, model_output)
            elif self.model_mean_type == ModelMeanType.RESIDUAL:
                pred_xstart_pre = self._predict_xstart_from_residual(
                    y=z_y, residual=model_output,
                )
                pred_eps = self._predict_eps_from_xstart(z_t, z_y, t, pred_xstart_pre)
            else:
                raise NotImplementedError(self.model_mean_type)
            
            grad, _ = self.cond_fn(
                x=z_t,
                y=z_y,
                t_=t,
                pred_noise=pred_eps.detach(),
                ground_truth=x_start,         # HR image in pixel space
                first_stage_model=first_stage_model,
            )

            noise_scale = _extract_into_tensor(
                self.kappa * self.sqrt_etas, t, pred_eps.shape
            )
            pred_eps_corr = pred_eps + self.rho * noise_scale * grad
            z0_corr = self._predict_xstart_from_eps(z_t, z_y, t, pred_eps_corr)

            if self.model_mean_type == ModelMeanType.EPSILON:
                effective_model_output = pred_eps_corr
            elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                scale = _extract_into_tensor(
                    self.kappa * self.sqrt_etas, t, pred_eps_corr.shape
                )
                effective_model_output = pred_eps_corr * scale
            elif self.model_mean_type == ModelMeanType.START_X:
                effective_model_output = z0_corr
            elif self.model_mean_type == ModelMeanType.RESIDUAL:
                effective_model_output = z_y - z0_corr
            else:
                raise NotImplementedError(self.model_mean_type)

            terms["guidance_grad_norm"] = grad.detach().pow(2).mean(
                dim=tuple(range(1, grad.ndim))
            ).sqrt()
        else:
            effective_model_output = model_output

        target = {
            ModelMeanType.START_X: z_start,
            ModelMeanType.RESIDUAL: z_y - z_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.EPSILON_SCALE: noise * self.kappa * _extract_into_tensor(
                self.sqrt_etas, t, noise.shape
            ),
        }[self.model_mean_type]
        assert effective_model_output.shape == target.shape

        if loss_in_image_space:
            assert self.model_mean_type == ModelMeanType.START_X
            model_output_rgb = self.decode_first_stage(
                effective_model_output, first_stage_model, no_grad=False,
            )
            target_rgb = self.decode_first_stage(z_start, first_stage_model)
            terms[loss_type] = mean_flat(
                (target_rgb - model_output_rgb) ** 2 if loss_type == "mse"
                else (target_rgb - model_output_rgb).abs()
            )
        else:
            terms[loss_type] = mean_flat(
                (target - effective_model_output) ** 2 if loss_type == "mse"
                else (target - effective_model_output).abs()
            )
        if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            terms[loss_type] /= (self.kappa ** 2 * _extract_into_tensor(self.etas, t, t.shape))
        weights = (
            _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            if self.loss_type == LossType.WEIGHTED_MSE else 1
        )
        terms["loss"] = terms["loss"] + terms[loss_type] * weights

        if learn_xT:
            predicted_xT = model(self._scale_input(z_start_teacher, t), t * 0, **model_kwargs)
            terms[loss_type + "_xT"] = mean_flat(
                (z_t - predicted_xT) ** 2 if loss_type == "mse"
                else (z_t - predicted_xT).abs()
            )
            terms["loss"] = terms["loss"] + terms[loss_type + "_xT"]

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = effective_model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(
                x_t=z_t, y=z_y, t=t, eps=effective_model_output.detach(),
            )
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(
                y=z_y, residual=effective_model_output.detach(),
            )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(
                x_t=z_t, y=z_y, t=t, eps=effective_model_output.detach(),
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        if finetune_use_gt:
            z_start_gt = self.encode_first_stage(x_start, first_stage_model, up_sample=False)
            if not xT_cov_loss:
                with th.no_grad():
                    predicted_xT_from_gt = model(
                        self._scale_input(z_start_gt, t), t * 0, **model_kwargs,
                    )
                    if False: 
                        noise_gt_pred = predicted_xT_from_gt - z_y
                        sampled_noise = z_t - z_y
                        noise_gt_new = noise_gt_pred / noise_gt_pred.std() * sampled_noise.std()
                        predicted_xT_from_gt = z_y + noise_gt_new
            else:
                predicted_xT_from_gt = model(
                    self._scale_input(z_start_gt, t), t * 0, **model_kwargs,
                )
                noise_gt_pred = predicted_xT_from_gt - z_y.detach()
                terms["mse_xT_cov"] = self.cov_loss(noise_gt_pred)
                terms["loss"] = terms["loss"] + (terms["mse_xT_cov"] * xT_cov_loss)

            model_output_pedict_gt = model(
                self._scale_input(predicted_xT_from_gt.detach(), t), t, **model_kwargs,
            )
            if not loss_in_image_space:
                terms[loss_type + "_gt"] = mean_flat(
                    (z_start_gt - model_output_pedict_gt) ** 2 if loss_type == "mse"
                    else (z_start_gt - model_output_pedict_gt).abs()
                )
                terms["loss"] = terms["loss"] + (terms[loss_type + "_gt"] * finetune_use_gt)
            else:
                model_output_pedict_gt_rgb = self.decode_first_stage(
                    model_output_pedict_gt, first_stage_model, no_grad=False,
                )
                terms[loss_type + "_gt_rgb"] = mean_flat(
                    (x_start - model_output_pedict_gt_rgb) ** 2 if loss_type == "mse"
                    else (x_start - model_output_pedict_gt_rgb).abs()
                )
                terms["loss"] = terms["loss"] + (terms[loss_type + "_gt_rgb"] * finetune_use_gt)

            if pred_zstart is None:
                pred_zstart = model_output_pedict_gt

        return terms, z_t, pred_zstart

    def cov_loss(self, noise):
        feat = noise
        kernel_size = 8
        b, c, h, w = feat.shape
        feat = feat.view(b * c, 1, h, w)
        feat_unfold = F.unfold(feat, kernel_size=kernel_size, stride=1)
        feat_flatten = feat_unfold.permute(0, 2, 1).contiguous()

        def batch_cov(points):
            B, N, D = points.size()
            mean = points.mean(dim=1).unsqueeze(1)
            diffs = (points - mean).reshape(B * N, D)
            prods = th.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
            bcov = prods.sum(dim=1) / (N - 1)
            return bcov

        cov = batch_cov(feat_flatten)
        target_cov = th.eye(cov.shape[1]).repeat([cov.shape[0], 1, 1]).to(cov.device) * (
            (self.kappa * self.sqrt_etas)[-1]
        ) ** 2
        return mean_flat((target_cov - cov) ** 2).view(b, c).sum(dim=1)

    def training_losses(
            self, model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None,
    ):
        """Plain (non-distillation) training losses; unchanged from upstream."""
        if model_kwargs is None:
            model_kwargs = {}
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        terms = {}
        model_output = model(self._scale_input(z_t, t), t, **model_kwargs)

        if self.loss_type in (LossType.MSE, LossType.WEIGHTED_MSE):
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise * self.kappa * _extract_into_tensor(
                    self.sqrt_etas, t, noise.shape,
                ),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa ** 2 * _extract_into_tensor(self.etas, t, t.shape))
            weights = (
                _extract_into_tensor(self.weight_loss_mse, t, t.shape)
                if self.loss_type == LossType.WEIGHTED_MSE else 1
            )
            terms["loss"] = terms["mse"] * weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(
                x_t=z_t, y=z_y, t=t, eps=model_output.detach(),
            )
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(
                y=z_y, residual=model_output.detach(),
            )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(
                x_t=z_t, y=z_y, t=t, eps=model_output.detach(),
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if not self.normalize_input:
            return inputs
        if self.latent_flag:
            std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1)
            return inputs / std
        inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
        return inputs / inputs_max

    def ddim_sample(
        self, model, ground_truth, x, y, t,
        clip_denoised=True, denoised_fn=None,
        model_kwargs=None, ddim_eta=0.0,
    ):
        out = self.p_mean_variance(
            model=model, x_t=x, y=y, t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = (
            pred_xstart * out["ddim_k"]
            + out["ddim_m"] * x
            + out["ddim_j"] * y
        )
        return {"sample": sample, "pred_xstart": pred_xstart, "pred_noise": out["pred_noise"]}

    def ddim_sample_loop(
        self, y, ground_truth, model,
        noise=None, first_stage_model=None,
        start_timesteps=None,
        clip_denoised=True, denoised_fn=None,
        model_kwargs=None, device=None,
        progress=False, ddim_eta=0.0, zT=None,
        apply_decoder=True, one_step=False,
    ):
        # print(f"[ddim_sample_loop] start, one_step={one_step}, num_timesteps={self.num_timesteps}")
        final = None
        yield_count = 0
        for sample in self.ddim_sample_loop_progressive(
            y=y, ground_truth=ground_truth, model=model,
            noise=noise, first_stage_model=first_stage_model,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device, progress=progress,
            ddim_eta=ddim_eta, zT=zT, one_step=one_step,
        ):
            final = sample
            yield_count += 1
            # print(f"[ddim_sample_loop] yield #{yield_count}, sample is None: {sample is None}")
        # print(f"[ddim_sample_loop] DONE, total yields={yield_count}, final is None: {final is None}")
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def ddim_sample_loop_progressive(
        self, y, ground_truth, model,
        noise=None, first_stage_model=None,
        clip_denoised=True, denoised_fn=None,
        model_kwargs=None, device=None,
        progress=False, ddim_eta=0.0,
        zT=None, one_step=False,
    ):
        # print(f"[progressive] start, one_step={one_step}, num_timesteps={self.num_timesteps}")
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_sample = self.prior_sample(z_y, noise) if zT is None else zT
        indices = list(range(self.num_timesteps))[::-1]
        # print(f"[progressive] indices={indices}")

        out = None
        for i in indices:
            # print(f"[progressive] iter i={i}")
            t = th.tensor([i] * z_y.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model=model, ground_truth=ground_truth,
                    x=z_sample, y=z_y, t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    ddim_eta=ddim_eta,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                z_sample = out["sample"]

        # Inference-time guidance: apply at the final step.
        if out is not None and self._guidance_active(
            th.tensor([self.num_timesteps - 1], device=device)
        ):
            t0 = th.zeros(z_sample.shape[0], device=device, dtype=th.long)
        # if self.use_guidance and out is not None:
        #     t0 = th.zeros(z_sample.shape[0], device=device, dtype=th.long)
            pred_noise = out["pred_noise"]
            with th.enable_grad():
                scaled_gradient, _ = self.cond_fn(
                    z_sample, z_y, t0, pred_noise, ground_truth,
                    first_stage_model=first_stage_model,
                )
            guided_pred_noise = pred_noise + self.rho * scaled_gradient
            x_start_guided = self._predict_xstart_from_eps(
                x_t=z_sample, y=z_y, t=t0, eps=guided_pred_noise,
            )
            out["sample"] = x_start_guided
            out["pred_xstart"] = x_start_guided
            out["pred_noise"] = guided_pred_noise
            yield out
        elif out is not None:
            yield out

class GaussianDiffusionDDPM:
    def __init__(self, *, betas, model_mean_type, model_var_type,
                 scale_factor=None, sf=4):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.scale_factor = scale_factor
        self.sf = sf

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape,
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape,
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t,
                        clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarTypeDDPM.LEARNED, ModelVarTypeDDPM.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarTypeDDPM.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarTypeDDPM.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarTypeDDPM.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output),
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output),
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t,
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape,
            ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(self, model, x, t,
                 clip_denoised=True, denoised_fn=None, model_kwargs=None):
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape,
                      noise=None, clip_denoised=True, denoised_fn=None,
                      first_stage_model=None, model_kwargs=None,
                      device=None, progress=False):
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            model_kwargs=model_kwargs, device=device, progress=progress,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def p_sample_loop_progressive(self, model, shape,
                                  noise=None, clip_denoised=True, denoised_fn=None,
                                  model_kwargs=None, device=None, progress=False):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model, img, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(self, model, ground_truth, x, t,
                    clip_denoised=True, denoised_fn=None,
                    model_kwargs=None, eta=0.0):
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(self, model, x, t,
                            clip_denoised=True, denoised_fn=None,
                            model_kwargs=None, eta=0.0):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None, first_stage_model=None,
                         clip_denoised=True, denoised_fn=None,
                         model_kwargs=None, device=None,
                         progress=False, eta=0.0):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device, progress=progress, eta=eta,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def ddim_sample_loop_progressive(self, model, shape, noise=None,
                                     clip_denoised=True, denoised_fn=None,
                                     model_kwargs=None, device=None,
                                     progress=False, eta=0.0):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device).long()
            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t,
                        first_stage_model=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        z_start = self.encode_first_stage(x_start, first_stage_model)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, t, noise=noise)

        terms = {}
        model_output = model(z_t, t, **model_kwargs)
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=z_start, x_t=z_t, t=t,
            )[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(
                x_t=z_t, t=t, eps=model_output.detach(),
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0,
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _scale_input(self, inputs, t):
        return inputs

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        with th.no_grad():
            z_sample = 1 / self.scale_factor * z_sample
            z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
            out = first_stage_model.decode(z_sample)
        return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode="bicubic")
        if first_stage_model is None:
            return y
        with th.no_grad():
            y = y.type(dtype=next(first_stage_model.parameters()).dtype)
            z_y = first_stage_model.encode(y)
            out = z_y * self.scale_factor
        return out.type(ori_dtype)