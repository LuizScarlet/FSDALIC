import os
import sys
import time
import random
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataloader import ImageDataset
from utils.optimizers import configure_optimizers
from utils.utils import *
from models import Cheng2020Attention_Adapt, ELIC_Adapt, MLICPlusPlus_Adapt
import torch.nn.functional as F
from inference import eval_model


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


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm):
    """Stage 1."""

    model.train()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    t_start = time.time()
    for i, d in enumerate(train_dataloader):

        d = d.to(device)
        optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss.update(out_criterion["loss"])
        bpp_loss.update(out_criterion["bpp_loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr.update(out_criterion["psnr"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])

        if i % 100 == 0 :
            torch.cuda.empty_cache()

    t = time.time()-t_start
    print(
        f"Train epoch {epoch}: "
        f"Loss: {loss.avg:.4f} |"
        f"PSNR: {psnr.avg:.3f} |"
        f"Bpp: {bpp_loss.avg:.4f} |"
        f"Y_Bpp: {y_bpp.avg:.4f} |"
        f"Z_Bpp: {z_bpp.avg:.4f} |"
        f'Time : {t:.2f} |'
    )
    return

def train_one_epoch_dec(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm):
    """Stage 2."""
    
    model.train()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    t_start = time.time()
    for i, d in enumerate(train_dataloader):

        d = d.to(device)
        optimizer.zero_grad()
        out_net = model.forward_round(d)
        out_criterion = criterion(out_net, d)
        out_criterion["lambdaD"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss.update(out_criterion["loss"])
        bpp_loss.update(out_criterion["bpp_loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr.update(out_criterion["psnr"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])

        if i % 100 == 0 :
            torch.cuda.empty_cache()

    t = time.time()-t_start
    print(
        f"Train epoch {epoch}: "
        f"Loss: {loss.avg:.4f} |"
        f"PSNR: {psnr.avg:.3f} |"
        f"Bpp: {bpp_loss.avg:.4f} |"
        f"Y_Bpp: {y_bpp.avg:.4f} |"
        f"Z_Bpp: {z_bpp.avg:.4f} |"
        f'Time : {t:.2f} |'
    )
    return


def eval_epoch(epoch, val_dataloader, model, criterion, factor=64):
    """Inference on the validation set."""

    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for d in val_dataloader:
            d = d.to(device)
            H, W = d.size(2), d.size(3)
            pad_h = 0
            pad_w = 0
            if H % factor != 0:
                pad_h = factor * (H // factor + 1) - H
            if W % factor != 0:
                pad_w = factor * (W // factor + 1) - W
            x_padded = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            out_net = model(x_padded)
            out_net["x_hat"] = F.pad(out_net["x_hat"], (0, -pad_w, 0, -pad_h))
            out_criterion = criterion(out_net, d)

            loss.update(out_criterion["loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion["psnr"])
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])

    print(
        f"Test  epoch {epoch}: "
        f"Loss: {loss.avg:.4f} |"
        f"PSNR: {psnr.avg:.3f} |"
        f"Bpp: {bpp_loss.avg:.4f} |"
        f"Y_Bpp: {y_bpp.avg:.4f} |"
        f"Z_Bpp: {z_bpp.avg:.4f} |"
    )

    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Adaptation script.")
    parser.add_argument("--lmbda", type=float, required=True, choices=[0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483], help="Pretrained rate-distortion parameters")
    parser.add_argument("--model_name", type=str, required=True, choices=['Cheng2020', 'ELIC', 'MLICpp'], help="Baseline models")
    parser.add_argument("--pretrained_weight", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset" )
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save checkpoints")

    parser.add_argument("--sample_num", type=int, default=25, help="Numbers of available training samples")
    parser.add_argument("--val_prop", type=float, default=0.2, help="Proportion for validation set")
    parser.add_argument("--threshold", type=int, default=30, help="Threshold to step lr scheduler")
    parser.add_argument("--lr_scheduler", type=list, default=[5e-4, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5])
    parser.add_argument("--max_epoch", type=int, default=750)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)

    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=float, default=192.1)
    args = parser.parse_args()
    return args


def main(argv):

    args = parse_args(argv)
    print(args)
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_images = [os.path.join(args.train_dataset, fname) for fname in os.listdir(args.train_dataset)]
    test_images = [os.path.join(args.test_dataset, fname) for fname in os.listdir(args.test_dataset)]

    random.shuffle(train_images)
    train_num = int(args.sample_num * (1 - args.val_prop))
    val_images = train_images[train_num:args.sample_num]
    train_images = train_images[:train_num]
                    
    train_dataset = ImageDataset(train_images, transform=train_transforms)
    val_dataset = ImageDataset(val_images, transform=test_transforms)
    test_dataset = ImageDataset(test_images, transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if args.model_name == 'Cheng2020':
        net = Cheng2020Attention_Adapt(N = 128 if args.lmbda in [0.0018, 0.0035, 0.0067] else 192)
        find_dec_adapter = lambda x: "adapter" in x and ("g_s.7" in x or "g_s.8" in x)
    elif args.model_name == 'ELIC':
        net = ELIC_Adapt()
        find_dec_adapter = lambda x: "adapter" in x and ("g_s.synthesis_transform.11" in x or "g_s.synthesis_transform.12" in x or "g_s.synthesis_transform.13" in x)
    elif args.model_name == 'MLICpp':
        net = MLICPlusPlus_Adapt()
        find_dec_adapter = lambda x: "adapter" in x and ("g_s.synthesis_transform.5" in x or "g_s.synthesis_transform.6" in x)
        
    # Load pretrained models
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
    net.set_adapter()
    net = net.to(device)

    for name, p in net.named_parameters():
        p.requires_grad = False
        if "adapter" in name:
            p.requires_grad = True
    optimizer = configure_optimizers(net)
    
    save_ckpt_path = os.path.join(args.save_dir, args.model_name + '_' + str(args.lmbda) + '.pth')
    best_loss = eval_epoch(-1, val_dataloader, net, criterion)
    lr_index = 0
    counter = 0


    print("TRAINING ADAPTERS ...")
    for epoch in range(args.max_epoch):

        lr = args.lr_scheduler[lr_index]
        print("Learning rate:", lr)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr

        train_one_epoch(net, criterion, train_dataloader, optimizer, epoch, args.clip_max_norm)
        loss = eval_epoch(epoch, val_dataloader, net, criterion)

        if loss < best_loss:
            print(f"epoch {epoch} is best now!")
            save_state = {}
            for name in net.state_dict():
                if "adapter" in name:
                    save_state.update({name:net.state_dict()[name]})
            torch.save(save_state, save_ckpt_path)
            best_loss = min(loss, best_loss)
            counter = 0
        else:
            counter += 1
            if counter >= args.threshold:
                counter = 0
                if lr_index < len(args.lr_scheduler)-1:
                    lr_index += 1

    net_dict = net.state_dict()
    net_keys = net_dict.keys()
    adapter_dict={}
    checkpoint = torch.load(save_ckpt_path)
    for k, v in checkpoint.items():
        if k in net_keys:
            adapter_dict[k]=v
    net_dict.update(adapter_dict)
    net.load_state_dict(net_dict)

    for name, p in net.named_parameters():
        p.requires_grad = False
        if find_dec_adapter(name):
            p.requires_grad = True
    optimizer_dec = configure_optimizers(net)


    print("FINETUNING DECODER ...")
    lr = args.lr_scheduler[0]
    for param_group in optimizer_dec.param_groups: 
        param_group['lr'] = lr

    for epoch in range(args.max_epoch):
        train_one_epoch_dec(net, criterion, train_dataloader, optimizer_dec, epoch, args.clip_max_norm)
        loss = eval_epoch(epoch, val_dataloader, net, criterion)
        if loss < best_loss:
            print(f"epoch {epoch} is best now!")
            save_state = {}
            for name in net.state_dict():
                if "adapter" in name:
                    save_state.update({name:net.state_dict()[name]})
            torch.save(save_state, save_ckpt_path)
            best_loss = min(loss, best_loss)

    
    print("EVALUATING TEST SET ...")
    net_dict = net.state_dict()
    net_keys = net_dict.keys()
    adapter_dict={}
    checkpoint = torch.load(save_ckpt_path)
    for k, v in checkpoint.items():
        if k in net_keys:
            adapter_dict[k]=v
    net_dict.update(adapter_dict)
    net.load_state_dict(net_dict)
    net.eval()


    print("After Adaptation:")
    eval_model(test_dataloader, net, criterion)


if __name__ == "__main__":
    main(sys.argv[1:])
