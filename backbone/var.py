from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model_head.arhead_gmm import ARHead_gmm
from module.adaln import AdaLNSelfAttn, AdaLNBeforeHead

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C

class VAR(nn.Module):
    def __init__(self, img_size=256, vae_stride=16, patch_size=1, vae_embed_dim=16,
                class_num=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                num_gaussians=1,
                head_width=1024,
                head_depth=1
                ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2

        # --------------------------------------------------------------------------
        self.num_gaussians = num_gaussians
        self.out_dim = 3*num_gaussians
        self.depth, self.embed_dim, self.num_heads = depth, embed_dim, num_heads
        self.head_width = head_width
        
        self.cond_drop_rate = cond_drop_rate
        
        self.patch_nums = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        
        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.token_embed_dim, self.embed_dim)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.embed_dim / 3)
        self.class_num = class_num
        self.uniform_prob = torch.full((1, class_num), fill_value=1.0 / class_num, dtype=torch.float32)
        self.class_emb = nn.Embedding(self.class_num + 1, self.embed_dim)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.embed_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.embed_dim)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.embed_dim)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.embed_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.embed_dim, 6*self.embed_dim)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.embed_dim, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.embed_dim, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        self.init_weights(init_adaln=0.5, init_adaln_gamma=1e-3, init_head=0.02, init_std=init_std)

        # 6. AR model head
        self.arhead = ARHead_gmm(num_gaussians, self.token_embed_dim, self.embed_dim, inner_ar_width=head_width, inner_ar_depth=head_depth)
        

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_, w_, c * p ** 2)
        return x  # [n, h, w, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def construct_multi_scale_features(self, imgs):
        # patchify and mask (drop) tokens
        features = self.patchify(imgs)

        # Generate multi-scale features
        f_hat = torch.zeros_like(features)
        multi_scale_inputs, multi_scale_outputs = [], []

        for i in range(len(self.patch_nums)):
            if i < len(self.patch_nums) - 1:
                pn = self.patch_nums[i]
                h = F.interpolate(features - f_hat, size=(pn, pn), mode='bicubic')

                f_hat += F.interpolate(h, size=features.shape[-2:], mode='bicubic')
                new_inp = F.interpolate(f_hat, size=(self.patch_nums[i+1], self.patch_nums[i+1]), mode='bicubic')
                multi_scale_inputs.append(new_inp.flatten(2))
            else:
                h = features - f_hat

            multi_scale_outputs.append(h.flatten(2))

        multi_scale_inputs = torch.cat(multi_scale_inputs, dim=2).permute(0, 2, 1)
        multi_scale_outputs = torch.cat(multi_scale_outputs, dim=2).permute(0, 2, 1)

        return multi_scale_inputs, multi_scale_outputs


    def forward(self, imgs, labels):
        inputs, outputs = self.construct_multi_scale_features(imgs)

        bg, ed = 0, self.L
        bsz = inputs.shape[0]
        labels = torch.where(torch.rand(bsz, device=labels.device) < self.cond_drop_rate, self.class_num, labels)
        sos = cond_BD = self.class_emb(labels)
        sos = sos.unsqueeze(1).expand(bsz, self.first_l, -1) + self.pos_start.expand(bsz, self.first_l, -1)
        
        z = torch.cat((sos, self.word_embed(inputs.float())), dim=1)
        z += self.lvl_embed(self.lvl_1L[:, :ed].expand(bsz, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                
        for i, b in enumerate(self.blocks):
            z = b(x=z, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, causal=False)
    
        z = z.reshape(-1, self.embed_dim)
        outputs = outputs.reshape(-1, self.token_embed_dim)
        loss = self.arhead(z, outputs)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):        
        if labels is None:
            labels = torch.multinomial(self.uniform_prob, num_samples=bsz, replacement=True).reshape(bsz)
        elif isinstance(labels, int):
            labels = torch.full((bsz,), fill_value=self.class_num if labels < 0 else labels, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((labels, torch.full_like(labels, fill_value=self.class_num)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*bsz, self.first_l, -1) + self.pos_start.expand(2*bsz, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(bsz, self.token_embed_dim, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            z = next_token_map
            for b in self.blocks:
                z = b(x=z, cond_BD=cond_BD_or_gss, attn_bias=None, causal=True)
            z = z.reshape(-1, self.embed_dim)
            h_BChw = self.arhead.sample(z, temperature=temperature, cfg=cfg)
            h_BChw = h_BChw.reshape(2*bsz, pn*pn, self.token_embed_dim)
            h_BChw = h_BChw[:bsz]
                                    
            h_BChw = h_BChw.transpose_(1, 2).reshape(bsz, self.token_embed_dim, pn, pn)
            f_hat, next_token_map = self.get_next_autoregressive_input(si, f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(bsz, self.token_embed_dim, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return f_hat
    
    def get_next_autoregressive_input(self, si, f_hat, h_BChw):        
        if si == self.num_stages_minus_1:
            f_hat += h_BChw
            return f_hat, h_BChw
        
        h = F.interpolate(h_BChw, size=(self.patch_nums[-1], self.patch_nums[-1]), mode='bicubic')
        f_hat = f_hat + h
        next_token_map = F.interpolate(f_hat, size=(self.patch_nums[si+1], self.patch_nums[si+1]), mode='bicubic')
        return f_hat, next_token_map
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.embed_dim / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.embed_dim:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.embed_dim].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)


def var_d16(**kwargs):
    depth = 16
    num_heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24

    model = VAR(
        depth=depth, embed_dim=width, num_heads=num_heads, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        head_width=width,
        head_depth=1, **kwargs)
    return model
