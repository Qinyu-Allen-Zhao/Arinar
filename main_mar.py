import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

from huggingface_hub import upload_file
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder, CachedNpzData, CachedH5FolderDev

from models.vae import AutoencoderKL
from models import mar, var
from engine_mar import train_one_epoch, evaluate
import copy


def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=1, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=128, help='generation batch size')
    # FastMAR
    parser.add_argument('--diff_upper_steps', default=25, type=int)
    parser.add_argument('--diff_lower_steps', default=5, type=int)
    parser.add_argument('--diff_sampling_strategy', default="linear", type=str,
                        help='Diffusion sampling strategy')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--bf16', action='store_true',
                        help='use bf16 precision instead of fp16')
    
    # First layer AR parameters
    ## VAR params

    ## MAR params
    parser.add_argument('--enc_dec_depth', type=int, default=-1,
                        help='Encoder/Decoder depth')
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    parser.add_argument('--head_batch_mul', type=int, default=1)
    # Second layer AR parameters
    parser.add_argument('--head_type', type=str, 
                        # choices=['ar_gmm', 'ar_diff_loss', 'gmm_wo_ar', 'gmm_cov_wo_ar', 'ar_byte'], 
                        default='ar_gmm', help='head type (default: ar_gmm)')
    parser.add_argument('--num_gaussians', type=int, default=1)
    parser.add_argument('--inner_ar_width', type=int, default=1024)
    parser.add_argument('--inner_ar_depth', type=int, default=1)
    parser.add_argument('--head_width', type=int, default=1024)
    parser.add_argument('--head_depth', type=int, default=6)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--feature_group', type=int, default=1)
    parser.add_argument('--bilevel_schedule', default="constant", type=str,
                         help='use bilevel schedule for model head')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--huggingface_dir', default="QinyuZhao1116/Arinar", type=str,
                        help='HuggingFace directory')
    parser.add_argument('--huggingface_token', default=None, type=str,
                        help='HuggingFace token')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not args.evaluate:
        if args.use_cached:
            if os.path.exists(os.path.join(args.cached_path, "mar_cache.npz")):
                dataset_train = CachedNpzData(args.cached_path)
            elif os.path.exists(os.path.join(args.cached_path, "latent_cache.h5")):
                dataset_train = CachedH5FolderDev(args.cached_path)
            else:
                dataset_train = CachedFolder(args.cached_path)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # define the vae and mar model
    vae_kwargs = {
        "attn_resolutions": (16,)
    } if "ldm" in args.vae_path or "vavae" in args.vae_path else {}
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path, **vae_kwargs).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False
    if 'mar_cache' in args.cached_path:
        latent_mean, latent_std = 0.0, 1. / 0.2325
    else:
        latent_stats = torch.load(os.path.join(args.cached_path, "latents_stats.pt"))
        latent_mean, latent_std = latent_stats['mean'].to(device), latent_stats['std'].to(device)

    kwargs = {
        "num_sampling_steps": args.num_sampling_steps,
        "diff_sampling_strategy": args.diff_sampling_strategy,
        "diff_upper_steps": args.diff_upper_steps,
        "diff_lower_steps": args.diff_lower_steps,
        "feature_group": args.feature_group,
        "bilevel_schedule": args.bilevel_schedule,
        "enc_dec_depth": args.enc_dec_depth,
    }
    if args.model.startswith('mar'):
        model = mar.__dict__[args.model](
            img_size=args.img_size,
            vae_stride=args.vae_stride,
            patch_size=args.patch_size,
            vae_embed_dim=args.vae_embed_dim,
            mask_ratio_min=args.mask_ratio_min,
            label_drop_prob=args.label_drop_prob,
            class_num=args.class_num,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            buffer_size=args.buffer_size,
            num_gaussians=args.num_gaussians,
            grad_checkpointing=args.grad_checkpointing,
            inner_ar_width=args.inner_ar_width,
            inner_ar_depth=args.inner_ar_depth,
            head_width=args.head_width,
            head_depth=args.head_depth,
            head_type=args.head_type,
            head_batch_mul=args.head_batch_mul,
            **kwargs
        )
    elif args.model.startswith('var'):
        model = var.__dict__[args.model](
            img_size=args.img_size,
            vae_stride=args.vae_stride,
            patch_size=args.patch_size,
            vae_embed_dim=args.vae_embed_dim,
            class_num=args.class_num,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            num_gaussians=args.num_gaussians
        )
    else:
        raise NotImplementedError("Model not implemented")
    
    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(args, model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # resume training
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, latent_mean, latent_std, 
                 ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=True)
        return

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae, latent_mean, latent_std,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, latent_mean, latent_std, 
                     ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, latent_mean, latent_std, 
                         ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # If this is the main process, upload checkpoint.pth to huggingface
    if misc.is_main_process() and args.huggingface_dir is not None and not args.evaluate:
        upload_file(
            path_or_fileobj=os.path.join(args.output_dir, "checkpoint-last.pth"),
            path_in_repo=os.path.join(args.output_dir, "checkpoint-last.pth"),
            repo_id=args.huggingface_dir,
            token=args.huggingface_token,
            repo_type="model"
        )

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
