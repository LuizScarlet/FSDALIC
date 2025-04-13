import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import ImageDataset
from utils.metrics import compute_metrics
from utils.utils import *
from torchvision import transforms
from models import Cheng2020Attention_Adapt, ELIC_Adapt, MLICPlusPlus_Adapt
import torch.nn.functional as F


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['y_bpp'] = torch.log(output['likelihoods']['y']).sum() / (-math.log(2) * num_pixels)
        out['z_bpp'] = torch.log(output['likelihoods']['z']).sum() / (-math.log(2) * num_pixels)
        out["bpp_loss"] = out['y_bpp'] + out['z_bpp']
        out["mse_loss"] = self.mse(output["x_hat"], target)

        out["lambdaD"] = self.lmbda * 255 ** 2 * out["mse_loss"]
        out["loss"] = out["bpp_loss"] + out["lambdaD"]

        out["psnr"] = -10 * (torch.log(out["mse_loss"]) / np.log(10))

        return out
    

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


def eval_model(test_dataloader, model, criterion, factor=64):
    """Quick inference on the test set."""

    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for img in test_dataloader:

            img = img.to(device)
            _, _, H, W = img.shape
            
            pad_h = 0
            pad_w = 0
            if H % factor != 0:
                pad_h = factor * (H // factor + 1) - H
            if W % factor != 0:
                pad_w = factor * (W // factor + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            out_net = model(img_pad)
            out_net["x_hat"] = F.pad(out_net["x_hat"], (0, -pad_w, 0, -pad_h))
            out_criterion = criterion(out_net, img)

            rec = torch2img(out_net["x_hat"])
            img = torch2img(img)
            p, m = compute_metrics(rec, img)

            loss.update(out_criterion["loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(p)
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])
    print(
        f"== Inference == "
        f"RD Loss: {loss.avg:.4f} |"
        f"PSNR: {psnr.avg:.3f} |"
        f"Bpp: {bpp_loss.avg:.4f} |"
        f"Y_Bpp: {y_bpp.avg:.4f} |"
        f"Z_Bpp: {z_bpp.avg:.4f} |"
    )
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Test dataset")
    parser.add_argument("--lmbda", type=float, required=True, choices=[0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483], help="Pretrained rate-distortion parameters")
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
    criterion = RateDistortionLoss(lmbda=args.lmbda)

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
    eval_model(test_dataloader, net, criterion)

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
    eval_model(test_dataloader, net, criterion)


if __name__ == "__main__":
    main(sys.argv[1:])
