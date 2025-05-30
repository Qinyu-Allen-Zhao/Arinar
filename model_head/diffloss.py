import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion
from module.cond_mlp import SimpleMLPAdaLN
from diffusion.dpm_solver import NoiseScheduleVP, DPM_Solver


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, token_embed_dim, decoder_embed_dim, head_depth, head_width, 
                 grad_checkpointing=False, head_batch_mul=1,
                 diff_upper_steps=25, diff_lower_steps=5, diff_sampling_strategy="linear"):
        super(DiffLoss, self).__init__()
        self.in_channels = token_embed_dim
        self.net = SimpleMLPAdaLN(
            in_channels=token_embed_dim,
            model_channels=head_width,
            out_channels=token_embed_dim * 2,  # for vlb loss
            z_channels=decoder_embed_dim,
            num_res_blocks=head_depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.sampler = "default"
        if self.sampler == "DDIM":
            self.gen_diffusion = [
                create_diffusion(timestep_respacing="ddim"+str(step), noise_schedule="cosine")
                for step in range(diff_lower_steps, diff_upper_steps+1, 1)
            ]
        elif self.sampler == "DPM_Solver":
            betas = torch.from_numpy(self.train_diffusion.betas)
            self.noise_schedule = NoiseScheduleVP(betas=betas, schedule='discrete')
            self.solver = DPM_Solver(
                model_fn=self.dpm_forward,
                noise_schedule=self.noise_schedule,
                algorithm_type="dpmsolver",
            )
        else:
            self.gen_diffusion = [
                create_diffusion(timestep_respacing=str(step), noise_schedule="cosine")
                for step in range(diff_lower_steps, diff_upper_steps+1, 1)
            ]
        
        self.diff_sampling_strategy = diff_sampling_strategy
        self.head_batch_mul = head_batch_mul

    def dpm_forward(self, x, t_continuous, condition):
        t = (t_continuous * 1000)
        return self.net(x, t, condition)

    def forward(self, target, z, mask=None):
        target = target.repeat(self.head_batch_mul, 1)
        z = z.repeat(self.head_batch_mul, 1)
        mask = mask.repeat(self.head_batch_mul) if mask is not None else None

        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0, step=0, ar_num_iter=64):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        upper_step = ar_num_iter - 1
        if self.diff_sampling_strategy == "linear":
            schedule_id = int((upper_step-step) / upper_step * (len(self.gen_diffusion)-1))
        elif self.diff_sampling_strategy == "cosine":
            schedule_id = int((math.cos(math.pi * step / upper_step) + 1) / 2 * (len(self.gen_diffusion)-1))
        elif self.diff_sampling_strategy == "constant":
            schedule_id = 0 # Constant schedule
        elif self.diff_sampling_strategy == "two-stage":
            schedule_id = len(self.gen_diffusion) - 1 if step < upper_step / 2 else 0
        else:
            raise ValueError(f"Unknown sampling strategy: {self.diff_sampling_strategy}")
        # schedule_id = len(self.gen_diffusion) - 1 - schedule_id
        if self.sampler == "DDIM":
            sampled_token_latent = self.gen_diffusion[schedule_id].ddim_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False
            )
        elif self.sampler == "DPM_Solver":
            sampled_token_latent = self.solver.sample(
                    noise, z, steps=50, t_start=None, t_end=None, order=2, skip_type='time_uniform',
                    method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
                    atol=0.0078, rtol=0.05, return_intermediate=False,)
        else:
            sampled_token_latent = self.gen_diffusion[schedule_id].p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature
            )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift
