import torch
import torch.nn as nn
import torch.optim as optim


def configure_optimizers(net):
    parameters = {n for n, p in net.named_parameters() if p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4)
    return optimizer