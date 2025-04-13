import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class IdenticalConv1x1(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        gamma = torch.eye(in_ch)
        gamma = gamma.reshape(in_ch, in_ch, 1, 1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.zeros(in_ch)
        self.beta = nn.Parameter(beta)
    def forward(self, x): 
        out = F.conv2d(x, self.gamma, self.beta)
        return out


class Conv_Adapter(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.adapter1 = IdenticalConv1x1(in_ch)

    def forward(self, x):
        out = self.adapter1(x)
        return out
    

def LoRA_forward(layer, temp, A, B, padding=0, stride=1):
    weight_delta = ((B.clone() @ A.clone())).unsqueeze(2).unsqueeze(3)
    return F.conv2d(temp, layer.weight + weight_delta, layer.bias, padding=padding, stride=stride)
