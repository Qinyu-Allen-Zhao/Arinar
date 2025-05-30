import math
from math import pi
import pdb
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from module.adaln import AdaLNSelfAttn, AdaLNBeforeHead


class ByteConverter():
    def __init__(self):
        self.shifts = torch.tensor([31, 23, 12, 0]).cuda()
        self.bitwise_constants = torch.tensor([[0b1, 0b11111111, 0b11111111111, 0b111111111111]]).cuda()

    def feature2byte(self, x, lower_exp=122, upper_exp=131):
        int_repr = x.view(torch.int32).unsqueeze(-1)
        bytes_tensor = ((int_repr >> self.shifts) & self.bitwise_constants).to(torch.int64)
        bytes_tensor[..., 1] = bytes_tensor[..., 1].clip(lower_exp, upper_exp) - lower_exp
        bytes_tensor[..., 1] += bytes_tensor[..., 0] * (upper_exp - lower_exp + 1)
        
        return bytes_tensor[..., 1:]
    
    def byte2feature(self, inp, lower_exp=122, upper_exp=131):
        bytes_tensor = inp.clone()
        
        sign = bytes_tensor[..., [0]] // (upper_exp - lower_exp + 1)
        bytes_tensor[..., 0] = bytes_tensor[..., 0] % (upper_exp - lower_exp + 1) + lower_exp
        bytes_tensor = torch.cat([sign, bytes_tensor], dim=-1)

        x = (bytes_tensor << self.shifts).sum(-1)
        x = x.to(torch.int32).view(torch.float32)

        x[bytes_tensor[..., 1] == upper_exp] = x[bytes_tensor[..., 1] == upper_exp].sign() * 16.0

        return x


class ARHead_byte(nn.Module):
    def __init__(self, num_bytes, token_embed_dim, decoder_embed_dim, 
                 inner_ar_width=768, inner_ar_depth=1, head_width=768, head_depth=1):
        super(ARHead_byte, self).__init__()
        self.num_bytes = num_bytes
        self.token_embed_dim = token_embed_dim
        self.inner_ar_width = inner_ar_width
        
        # Input projection
        self.vocabulary_size = [20, 2048, 4096]
        self.word_embed = nn.ModuleList([nn.Embedding(self.vocabulary_size[i], self.inner_ar_width).cuda() 
                                         for i in range(num_bytes)])
        self.cond_proj = nn.Linear(decoder_embed_dim, inner_ar_width)

        # Start token and position embedding
        self.start_token = nn.Parameter(torch.empty(1, 1, inner_ar_width))
        self.pos_embedding = nn.Parameter(torch.empty(1, token_embed_dim * num_bytes, inner_ar_width))
        self.level_embed = nn.Embedding(num_bytes, inner_ar_width)

        # Backbone blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.drop_path_rate = 0.
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, inner_ar_depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=decoder_embed_dim,
                block_idx=block_idx, embed_dim=inner_ar_width, norm_layer=norm_layer, num_heads=16, mlp_ratio=4.,
                drop=0., attn_drop=0., drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=False, shared_aln=False,
                flash_if_available=True, fused_if_available=True,
            )
            for block_idx in range(inner_ar_depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # Model head
        self.head_nm = nn.ModuleList([nn.LayerNorm(inner_ar_width, eps=1e-6, elementwise_affine=False)
                                      for i in range(num_bytes)])
        self.head = nn.ModuleList([nn.Linear(inner_ar_width, self.vocabulary_size[i])
                                        for i in range(num_bytes)])

        self.init_weights()

        self.byte_converter = ByteConverter()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.loc_ids = torch.tensor(range(num_bytes), device="cuda").unsqueeze(0).repeat(1, self.token_embed_dim)
 
    def forward(self, z, target, mask=None):
        bsz = z.shape[0]

        # Convert target to byte representation
        target = self.byte_converter.feature2byte(target)  # [bsz, token_embed_dim, num_bytes]

        # Construct inputs
        inp = torch.split(target, 1, dim=-1)  # [bsz, token_embed_dim, 1] * num_bytes
        inp = [word_emb(x) for word_emb, x in zip(self.word_embed, inp)]
        inp = torch.cat(inp, dim=2)  # [bsz, token_embed_dim, num_bytes, inner_ar_width]
        inp = inp.reshape(bsz, self.token_embed_dim * self.num_bytes, self.inner_ar_width)

        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        level_pos_emb = self.level_embed(self.loc_ids).expand(bsz, -1, -1)
        x = torch.cat((start, inp[:, :-1]), dim=1) + level_pos_emb + self.pos_embedding.expand(bsz, -1, -1)

        for b in self.blocks:
            x = b(x=x, cond_BD=z, attn_bias=None, causal=True)

        total_loss = None
        for i in range(self.num_bytes):
            head_nm, head = self.head_nm[i], self.head[i]
            x_split = x[:, i::self.num_bytes]
            x_split = head(head_nm(x_split))

            x_split = x_split.reshape(-1, self.vocabulary_size[i])

            # Cross entropy loss
            loss = self.loss_func(x_split, target[:, :, i].flatten()).reshape(bsz, self.token_embed_dim)

            # Apply mask for super large or small values
            if i == 1:
                loss = loss.masked_fill(target[:, :, i-1] % 8 == 7, 0)

            loss = loss.mean(-1)
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
            
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
            
            print(f"Head {i}, x_split max: {x_split.max().item():.3f}, min: {x_split.min().item():.3f}, loss: {loss.item():.3f}")

        return total_loss / self.num_bytes

    def sample(self, z, temperature=1.0, cfg=1.0, top_p=0.99, **kwargs):
        bsz = z.shape[0]

        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        for b in self.blocks: b.attn.kv_caching(True)
        x = start
        res = []

        for i in range(self.token_embed_dim):
            for byte in range(self.num_bytes):
                x = x + self.level_embed(self.loc_ids[:, byte:byte+1]).expand(bsz, 1, -1)
                pos = i * self.num_bytes + byte
                x = x + self.pos_embedding[:, pos:pos+1].expand(bsz, 1, -1)

                for b in self.blocks:
                    x = b(x=x, cond_BD=z, attn_bias=None, causal=False)
                head, head_nm = self.head[byte], self.head_nm[byte]
                x = head(head_nm(x))

                x = x.squeeze().softmax(dim=-1)
                x = torch.multinomial(x, 1)

                res.append(x)

                x = self.word_embed[byte](x)
        
        for b in self.blocks: b.attn.kv_caching(False)
        res = torch.cat(res, dim=1)
        res = res.reshape(bsz, self.token_embed_dim, self.num_bytes)
        res = self.byte_converter.byte2feature(res)

        return res

    def init_weights(self, init_adaln=0.02, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        nn.init.trunc_normal_(self.start_token.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pos_embedding.data, mean=0, std=init_std)
        for word_emb in self.word_embed:
            nn.init.trunc_normal_(word_emb.weight.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.cond_proj.weight.data, mean=0, std=init_std)
        self.cond_proj.bias.data.zero_()
        nn.init.trunc_normal_(self.level_embed.weight.data, mean=0, std=init_std)
        
        if init_head >= 0:
            for head in self.head:
                if isinstance(head, nn.Linear):
                    head.weight.data.mul_(init_head)
                    head.bias.data.zero_()
                elif isinstance(head, nn.Sequential):
                    head[-1].weight.data.mul_(init_head)
                    head[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.inner_ar_width:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.inner_ar_width].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
