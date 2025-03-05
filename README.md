**ARINAR: Bi-Level Autoregressive Feature-by-Feature Generative Models** [[Paper](https://arxiv.org/abs/2503.02883)]

*Qinyu Zhao, Stephen Gould, Liang Zheng*

:fire: This repository contains the source code for the technical report. We will conduct more ablation and variant studies and scale up this model in our report updates. 

:+1: Our codebase heavily depends on [MAR](https://github.com/LTH14/mar) and [VAR](https://github.com/FoundationVision/VAR). We really appreciate their outstanding work.

:star: Besides, a recent work [FractalGen](https://github.com/LTH14/fractalgen) explores a similar idea. We encourage readers to refer to their paper for further insights.

## Abstract
Existing autoregressive (AR) image generative models use a token-by-token generation schema. That is, they predict a per-token probability distribution and sample the next token from that distribution. The main challenge is how to model the complex distribution of high-dimensional tokens. Previous methods either are too simplistic to fit the distribution or result in slow generation speed. Instead of fitting the distribution of the whole tokens, we explore using a AR model to generate each token in a feature-by-feature way, i.e., taking the generated features as input and generating the next feature. Based on that, we propose ARINAR (AR-in-AR), a bi-level AR model. The outer AR layer take previous tokens as input, predicts a condition vector $\boldsymbol{z}$ for the next token. The inner layer, conditional on $\boldsymbol{z}$, generates features of the next token autoregressively. In this way, the inner layer only needs to model the distribution of a single feature, for example, using a simple Gaussian Mixture Model. On the ImageNet 256x256 image generation task, ARINAR-B with 213M parameters achieves an FID of 2.75, which is comparable to the state-of-the-art MAR-B model (FID=2.31), while five times faster than the latter.

## Getting Started

### Installation
####  Clone this repository to your local machine.

```
git clone https://github.com/Qinyu-Allen-Zhao/Arinar.git
cd Arinar
conda env create -f environment.yaml
conda activate arinar
```

Then, please install [FlashAttention](https://github.com/Dao-AILab/flash-attention).
```
pip install flash-attn --no-build-isolation
```


### Training
#### 1. Prepare the Dataset
Please download and prepare the training set of [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index). 


#### 2. Download VAE Model Pre-Trained by [MAR](https://github.com/LTH14/mar)
```
python util/download.py
```

#### 3. (Optional) Caching VAE Latents, Following [MAR](https://github.com/LTH14/mar)
```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

#### 4. Start Training
```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
main_mar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --num_gaussians 4 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --output_dir ./outputs/mar_base_arinar_4_w768_d1 --resume ./outputs/mar_base_arinar_4_w768_d1 \
--data_path ${IMAGENET_PATH} --num_workers 2 --pin_mem \
--online_eval
```
Please add ```--use_cached --cached_path ${CACHED_PATH}``` if you want to train with cached VAE latents.

It takes about 8 days to train the base model on 4x A100(80G) GPUs, with cached VAE latents.


### Evaluation
```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
main_mar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --num_gaussians 4 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --output_dir ./outputs/mar_base_arinar_4_w768_d1 --resume ./outputs/mar_base_arinar_4_w768_d1 \
--data_path ${IMAGENET_PATH} --num_workers 2 --pin_mem \
--temperature 1.0 --cfg 4.5
--evaluate
```
When temperature = 1.1, the best cfg is 3.9.

The checkpoint we trained was uploaded to [GoogleDrive](https://drive.google.com/drive/folders/1Mq5SbG-oKPOwmaz3i-9d8PKpW12yPdRE?usp=sharing). Feel free to download and evaluate it.

| Model  | #Parameters      | FID  | Time / image (s) |
|---------| --------------|------|------------------|
| MAR-B | 208M | 2.31 | 65.69           |
| FractalMAR-B | 186M | 11.80 | 137.62          |
| ARINAR-B | 213M | 2.75 | 11.57           |


## Citation

If you use our codebase or our results in your research, please cite our work:

```bibtex
@article{zhao2025arinar,
  title={ARINAR: Bi-Level Autoregressive Feature-by-Feature Generative Models},
  author={Zhao, Qinyu and Gould, Stephen and Zheng, Liang},
  journal={arXiv preprint arXiv:2503.02883},
  year={2025}
}
```