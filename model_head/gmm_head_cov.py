import torch
import torch.nn as nn

from module.uncond_mlp import UncondSimpleMLP

class GMMCovHead(nn.Module):
    def __init__(self, num_gaussians, token_embed_dim, decoder_embed_dim, width=1024, depth=6, grad_checkpointing=False):
        super(GMMCovHead, self).__init__()
        self.num_gaussians = num_gaussians
        self.token_embed_dim = token_embed_dim
        self.num_cholesky_params = token_embed_dim * (token_embed_dim + 1) // 2 * num_gaussians
        self.output_size = self.token_embed_dim * self.num_gaussians + self.num_cholesky_params + self.num_gaussians  # mean, std, and weight of Gaussians
        self.tril_indices = torch.tril_indices(self.token_embed_dim, self.token_embed_dim, device="cuda")
        self.net = UncondSimpleMLP(
            in_channels=decoder_embed_dim,
            model_channels=width,
            out_channels=self.output_size,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

    def _build_lower_triangular(self, cholesky_params):
        """
        Convert a vector of size num_cholesky_params into a D x D lower
        triangular matrix, for each (batch_size, num_gaussians).
        
        Diagonal entries are exponentiated (plus a small epsilon) to ensure
        positive values on the diagonal for positive-definite covariance.
        """
        cholesky_params = cholesky_params.view(-1, self.num_gaussians, self.num_cholesky_params // self.num_gaussians)
        batch_size = cholesky_params.shape[0]
        
        L = torch.zeros(batch_size, self.num_gaussians, self.token_embed_dim, self.token_embed_dim, device=cholesky_params.device, dtype=cholesky_params.dtype)
        
        L[..., self.tril_indices[0], self.tril_indices[1]] = cholesky_params

        # Ensure positive diagonal elements for numerical stability
        diag_indices = torch.arange(self.token_embed_dim, device=cholesky_params.device)
        L[..., diag_indices, diag_indices] = torch.exp(L[..., diag_indices, diag_indices]).clamp(min=1e-5).to(L.dtype)
        
        return L

    def extract_gmm(self, pred):
        weight = pred[:, -self.num_gaussians:]
        weight = weight.softmax(dim=-1)

        mu = pred[:, : self.token_embed_dim * self.num_gaussians].reshape(-1, self.num_gaussians, self.token_embed_dim)

        cholesky_params = pred[:, self.token_embed_dim * self.num_gaussians: self.token_embed_dim * self.num_gaussians + self.num_cholesky_params]
        L = self._build_lower_triangular(cholesky_params)

        return weight, mu, L
    
    def forward(self, z, target, mask):
        pred = self.net(z)
        
        weight, mu, L = self.extract_gmm(pred)
        
        # Covariance: Sigma = L @ L^T, shape => (batch_size, K, D, D)
        # Sigma = L @ L.transpose(-1, -2)

        # Compute log-likelihood for each mixture component
        # log N(x | mu_k, Sigma_k)
        #   = -0.5 [D * log(2*pi) + log(det(Sigma_k)) + (x-mu_k) Sigma_k^{-1} (x-mu_k)^T]
        # We'll do this carefully for stability and efficiency.
        
        # 1) log(det(Sigma_k)) = 2 * sum(log(diag(L_k)))
        # because Sigma_k = L_k L_k^T, det(Sigma_k) = (prod(diag(L_k)))^2
        diag_L = L.diagonal(dim1=-2, dim2=-1)  # shape: (batch_size, K, D)
        log_det_Sigma = 2.0 * torch.sum(torch.log(diag_L), dim=-1)  # (batch_size, K)
        
        # 2) (x - mu)
        diff = target.unsqueeze(1) - mu  # shape => (batch_size, K, D)
        
        # 3) Solve for ((x-mu_k)^TSigma_k^{-1}(x-mu_k)):
        # Solve L @ y = diff for y, where L is the Cholesky factor
        # This is equivalent to computing (L^{-1}) @ diff efficiently
        y = torch.einsum('bkdd,bkd->bkd', torch.inverse(L.float()), diff)

        # Quadratic form: shape => (batch_size, K)
        quad_form = torch.sum(y**2, dim=-1)
        
        # Putting it all together:
        # log N(x | mu_k, Sigma_k) = -0.5 [D*log(2*pi) + log_det_Sigma + quad_form]
        log_prob_x_given_k = -0.5 * (29.40603 + log_det_Sigma + quad_form)  # (batch_size, K)
        
        # Mixture weighting:
        # log [ sum_k pi_k * N(x|mu_k, Sigma_k) ] = logsumexp( log_pi_k + log N(x|mu_k) , dim=k )
        log_pi = torch.log(weight)
        log_mixture = torch.logsumexp(log_pi + log_prob_x_given_k, dim=-1)  # (batch_size,)
        
        # Negative log-likelihood
        nll = -log_mixture
        nll = (nll * mask).sum() / mask.sum()

        # Probability of the target
        x0 = self.sample_from_gmm(weight, mu, L).reshape(-1, 256, 16)
        x1 = target.reshape(-1, 256, 16)
        t = torch.rand_like(x0)
        xt = t * x1 + (1-t) * x0
        z = z.reshape(-1, 256, z.shape[-1])

        velocity = self.flow_net(xt, z)

        y = x1 - x0
        rec_loss = (velocity - y).pow(2).mean(dim=-1).flatten()
        if mask is not None:
            rec_loss = (rec_loss * mask).sum() / mask.sum()

        return nll + rec_loss

    def sample(self, z, num_steps=100, temperature=1.0, cfg=1.0, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            pred = self.net(z)
        weight, mu, L = self.extract_gmm(pred)
        x0 = self.sample_from_gmm(weight, mu, L)

        return x0

    def sample_from_gmm(self, weight, mu, L):
        # Sample from the Gaussian Mixture Model
        mixture = torch.distributions.Categorical(weight)
        idx = mixture.sample()
        mvn = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        sample = mvn.sample()
        sample = sample[torch.arange(sample.size(0)), idx]

        return sample
