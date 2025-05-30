from math import pi
import torch
import torch.nn as nn

from module.uncond_mlp import UncondSimpleMLP


class GMMHead(nn.Module):
    def __init__(self, num_gaussians, token_embed_dim, decoder_embed_dim, width=1024, depth=6, grad_checkpointing=False):
        super(GMMHead, self).__init__()
        self.num_gaussians = num_gaussians
        self.token_embed_dim = token_embed_dim
        self.output_size = self.token_embed_dim * self.num_gaussians * 2 + self.num_gaussians  # mean, std, and weight of Gaussians
        self.gmm_predictor = UncondSimpleMLP(
            in_channels=decoder_embed_dim,
            model_channels=width,
            out_channels=self.output_size,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

    def extract_gmm(self, pred):
        weight = pred[:, -self.num_gaussians:]
        weight = weight.softmax(dim=-1)

        mu = pred[:, : self.token_embed_dim * self.num_gaussians].reshape(-1, self.num_gaussians, self.token_embed_dim)

        var = pred[:, self.token_embed_dim * self.num_gaussians: 2 * self.token_embed_dim * self.num_gaussians]
        var = var.reshape(-1, self.num_gaussians, self.token_embed_dim)
        var = nn.functional.softplus(var).clamp(min=1e-6)
        return weight, mu, var
    
    def forward(self, z, target, mask=None):
        pred = self.gmm_predictor(z)

        weight, mu, var = self.extract_gmm(pred)
        
        # Multi-variate Gaussian likelihood
        diff = target.unsqueeze(1) - mu  # [bsz*seq_len, num_gaussians, token_embed_dim]
        const_term = torch.ones_like(mu) * 2 * pi
        log_likelihood = -0.5 * (diff**2 / var + torch.log(var) + torch.log(const_term))  # [bsz*seq_len, num_gaussians, token_embed_dim]
        log_likelihood = log_likelihood.sum(dim=-1)  # [bsz*seq_len, num_gaussians]
        log_likelihood = torch.logsumexp(torch.log(weight) + log_likelihood, dim=-1)  # [bsz*seq_len]
        nll = -log_likelihood # Calculate NLL loss
        if mask is not None:
            nll = (nll * mask).sum() / mask.sum()

        return nll

    def sample(self, z, temperature=1.0, cfg=1.0, **kwargs):
        assert temperature == 1.0, "Temperature is not supported for GMM sampling."
        assert cfg == 1.0, "CFG is not supported for GMM sampling."

        with torch.cuda.amp.autocast(dtype=torch.float32):
            pred = self.gmm_predictor(z)
        weight, mu, var = self.extract_gmm(pred)
        x = self.sample_from_gmm(weight, mu, var)

        return x
    
    def sample_from_gmm(self, weight, mu, var):
        # Sample from the Gaussian Mixture Model
        mixture = torch.distributions.Categorical(weight)
        idx = mixture.sample()

        eps = torch.randn_like(mu)
        sample = eps * torch.sqrt(var) + mu  # [bsz*seq_len, num_gaussians, token_embed_dim]

        sample = sample[torch.arange(sample.size(0)), idx]

        return sample
    
