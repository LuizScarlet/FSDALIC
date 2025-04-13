import os
import sys
import math
import time
import logging
from logging import handlers
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloader import ImageDataset
from utils.metrics import compute_metrics
from utils.utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Cheng2020Attention_Adapt, ELIC_Adapt, MLICPlusPlus_Adapt
    

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='W0', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.handlers.TimedRotatingFileHandler(
            filename=filename, when=when, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def compress_one_image(model, x, stream_path, H, W, img_name):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    if not os.path.exists(stream_path): os.makedirs(stream_path)
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    enc_time = end_time - start_time
    return bpp, enc_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return x_hat, dec_time


def test_model(test_dataloader, model, log, save_dir, factor=64):
    """Practical entropy coding on the test set."""
    
    device = next(model.parameters()).device
    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()
    stream_path = os.path.join(save_dir, 'bin')
    gt_path = os.path.join(save_dir, 'gt')
    rec_path = os.path.join(save_dir, 'rec')

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            
            img = img.to(device)
            _, _, H, W = img.shape

            pad_h = 0
            pad_w = 0
            if H % factor != 0:
                pad_h = factor * (H // factor + 1) - H
            if W % factor != 0:
                pad_w = factor * (W // factor + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            bpp, enc_time = compress_one_image(model=model, x=img_pad, stream_path=stream_path, H=H, W=W, img_name=str(1))
            x_hat, dec_time = decompress_one_image(model=model, stream_path=stream_path, img_name=str(1))
            
            rec = torch2img(x_hat)
            img = torch2img(img)
            # img.save(os.path.join(gt_path, str(i) + '_gt.png'))
            # rec.save(os.path.join(rec_path, str(i) + '_rec.png'))
            p, m = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)

            if i >=3: # Warm up GPU
                avg_deocde_time.update(dec_time)
                avg_encode_time.update(enc_time)

            log.logger.info(
                f"IMG ID: {i:d} | "
                f"PSNR: {p:.7f} | "
                f"Bpp: {bpp:.7f} | "
                f"MS-SSIM: {m:.7f} | "
                f"Avg Encoding Latency: {enc_time:.6f} | "
                f"Avg Decoding latency: {enc_time:.6f}"
            )

    log.logger.info(f"====== Finish Testing {i+1:d} IMGs ======")
    log.logger.info(
        f"PSNR: {avg_psnr.avg:.7f} | "
        f"Bpp: {avg_bpp.avg:.7f} | "
        f"MS-SSIM: {avg_ms_ssim.avg:.7f} | "
        f"Avg Encoding Latency: {avg_encode_time.avg:.6f} | "
        f"Avg Decoding latency: {avg_deocde_time.avg:.6f}"
    )
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Practical entropy coding script.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Test dataset")
    parser.add_argument("--lmbda", type=float, required=True, choices=[0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483], help="Pretrained rate-distortion parameters")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save bitstreams and reconstructions.")
    parser.add_argument("--model_name", type=str, required=True, choices=['Cheng2020', 'ELIC', 'MLICpp'], help="Baseline models")
    parser.add_argument("--pretrained_weight", type=str, required=True)
    parser.add_argument("--adapters_weight", type=str, required=True)

    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_images = [os.path.join(args.test_dataset, fname) for fname in os.listdir(args.test_dataset)]
    test_dataset = ImageDataset(test_images, transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if args.model_name == 'Cheng2020':
        net = Cheng2020Attention_Adapt(N = 128 if args.lmbda in [0.0018, 0.0035, 0.0067] else 192)
    elif args.model_name == 'ELIC':
        net = ELIC_Adapt()
    elif args.model_name == 'MLICpp':
        net = MLICPlusPlus_Adapt()

    log = Logger(filename=os.path.join(args.save_dir, 'test.log'), level='info', fmt="%(asctime)s - %(message)s")
        
    # Load and evaluate pretrained models on out-of-domain images
    net_dict = net.state_dict()
    net_keys = net_dict.keys()
    pretrained_dict={}
    checkpoint = torch.load(args.pretrained_weight)
    for k, v in checkpoint.items():
        if k in net_keys:
            pretrained_dict[k]=v
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    net.eval()
    net = net.to(device)
    print("Before Adaptation:")
    net.update(force=True)
    test_model(test_dataloader, net, log, args.save_dir)

    # Load adapters and evaluate
    net.set_adapter()
    net_dict = net.state_dict()
    net_keys = net_dict.keys()
    adapter_dict={}
    checkpoint = torch.load(args.adapters_weight)
    for k, v in checkpoint.items():
        if k in net_keys:
            adapter_dict[k]=v
    net_dict.update(adapter_dict)
    net.load_state_dict(net_dict)
    net.eval()
    net = net.to(device)
    print("After Adaptation:")
    net.update(force=True)
    test_model(test_dataloader, net, log, args.save_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
