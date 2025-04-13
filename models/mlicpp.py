import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from modules.layers.adapter import Conv_Adapter, LoRA_forward
from torch.nn.init import trunc_normal_
from utils.func import update_registered_buffers, get_scale_table
from compressai.layers import conv3x3, subpel_conv3x3, GDN
from utils.ckbd import *


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def build_position_index(window_size):
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def model_config():
    config = Config({
        # MLIC and MLIC+
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "act": nn.GELU,
    })

    return config


class ResidualBlock(nn.Module):
    """ResidualBlock in MLIC++ with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None
        self.use_adapter = False
        self.channel = out_ch

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class ResidualBlockWithStride(nn.Module):
    """ResidualBlockWithStride in MLIC++ with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None
        self.use_adapter = False
        self.channel = out_ch

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class ResidualBlockUpsample(nn.Module):
    """ResidualBlockUpsample in MLIC++ with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.act = nn.GELU()
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.use_adapter = False
        self.channel = out_ch

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.act(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class AnalysisTransform(nn.Module):
    """g_a in MLIC++ with optional domain adapters."""

    def __init__(self, N, M):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2)
        )

    def set_adapter(self):
        """Construct domain adapters."""

        # set Conv_Adapters
        for module in self.analysis_transform:
            if hasattr(module, "set_adapter"):
                module.set_adapter()

    def forward(self, x):
        x = self.analysis_transform(x)
        return x
    

class SynthesisTransform(nn.Module):
    """g_s in MLIC++ with optional domain adapters."""

    def __init__(self, N, M):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def set_adapter(self):
        """Construct domain adapters."""

        # set Conv_Adapters
        for module in self.synthesis_transform:
            if hasattr(module, "set_adapter"):
                module.set_adapter()

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
    

class HyperAnalysis(nn.Module):
    """h_a in MLIC++."""

    def __init__(self, M=192, N=192):
        super().__init__()
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    

class HyperSynthesis(nn.Module):
    """h_s in MLIC++."""

    def __init__(self, M=192, N=192):
        super().__init__()
        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
    

class LocalContext(nn.Module):
    """Context module in MLIC++."""

    def __init__(self,
                 dim=32,
                 window_size=5,
                 mlp_ratio=2.,
                 num_heads=2,
                 qkv_bias=True,
                 qk_scale=None
                ) -> None:
        super().__init__()
        self.H = -1
        self.W = -1
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=window_size, stride=1, padding=(window_size - 1) // 2)
        self.relative_position_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim * 2, dim * 2)
        self.mlp = MLP(in_dim=dim * 2, hidden_dim=int(dim * 2 * mlp_ratio), out_dim=dim * 2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.register_buffer("relative_position_index", build_position_index((window_size, window_size)))
        self.attn_mask = None
        self.fusion = nn.Conv2d(dim, dim * 2, kernel_size=window_size)

    def update_resolution(self, H, W, device, mask=None):
        updated=False
        if self.H != H or self.W != W:
            self.H = H
            self.W = W
            if mask is not None:
                self.attn_mask = mask.to(device)
                updated=True
                return updated
            ckbd = torch.zeros((1, 2, H, W), device=device, requires_grad=False)
            # anchor
            ckbd[:, :, 0::2, 1::2] = 1
            ckbd[:, :, 1::2, 0::2] = 1
            qk_windows = self.unfold(ckbd).permute(0, 2, 1)
            qk_windows = qk_windows.view(1, H * W, 2, 1, self.window_size, self.window_size).permute(2, 0, 1, 3, 4, 5)
            q_windows, k_windows = qk_windows[0], qk_windows[1]
            q = q_windows.reshape(1, H * W, 1, self.window_size * self.window_size).permute(0, 1, 3, 2)
            k = k_windows.reshape(1, H * W, 1, self.window_size * self.window_size).permute(0, 1, 3, 2)
            attn_mask = (q @ k.transpose(-2, -1))
            attn_mask = attn_mask.masked_fill(attn_mask == 0., float(-100.0)).masked_fill(attn_mask == 1, float(0.0))
            self.attn_mask = attn_mask[0].detach()
            updated=True
        return updated

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        self.update_resolution(H, W, x.device)
        # [B, L, C]
        x = x.reshape(B, C, L).permute(0, 2, 1)
        x = self.norm1(x)

        # [3, B, C, H, W]
        qkv = self.qkv_proj(x).reshape(B, H, W, 3, C).permute(3, 0, 4, 1, 2)

        # window partition
        q, k, v = qkv[0], qkv[1], qkv[2]
        qkv = torch.cat([q, k, v], dim=1)
        qkv_windows = self.unfold(qkv).permute(0, 2, 1)
        qkv_windows = qkv_windows.view(B, L, 3, C, self.window_size, self.window_size).permute(2, 0, 1, 3, 4, 5)
        # [B, L, C, window_size, window_size]
        q_windows, k_windows, v_windows = qkv_windows[0], qkv_windows[1], qkv_windows[2]

        # [B, L, num_heads, window_size * window_size, head_dim]
        q = q_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)
        k = k_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)
        v = v_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)

        q = q * self.scale
        # [B, L, num_heads, window_size * window_size, window_size * window_size]
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        # [num_heads, window_size * window_size, window_size * window_size]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(1)

        attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(2)

        attn = self.softmax(attn)

        x = (attn @ v).reshape(B, L, self.num_heads, self.window_size, self.window_size, self.head_dim).permute(0, 1, 3, 4, 2, 5)
        x = x.reshape(B * L, self.window_size, self.window_size, C).permute(0, 3, 1, 2)
        x = self.fusion(x).reshape(B, L, C * 2)
        x = self.proj(x)
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).reshape(B, C * 2, H, W)
        return x


class ChannelContext(nn.Module):
    """Context module in MLIC++."""

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 192, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim * 4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        channel_params = self.fushion(channel_params)
        return channel_params
    

class LinearGlobalIntraContext(nn.Module):
    """Context module in MLIC++."""

    def __init__(
            self,
            dim=32,
            num_heads=2) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.reprojection = nn.Conv2d(dim, dim * 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1)
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_ac = ckbd_anchor(x1)
        x1_na = ckbd_nonanchor(x1)
        queries = ckbd_nonanchor_sequeeze(self.queries(x1_na)).reshape(B, self.dim, H * W//2)
        keys = ckbd_anchor_sequeeze(self.keys(x1_ac)).reshape(B, self.dim, H * W//2)
        values = ckbd_anchor_sequeeze(self.values(x2)).reshape(B, self.dim, H * W//2)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(keys[:, i * head_dim: (i + 1) * head_dim, :], dim=2)
            query = F.softmax(queries[:, i * head_dim: (i + 1) * head_dim, :], dim=1)
            value = values[:, i * head_dim: (i + 1) * head_dim, :]
            key = ckbd_anchor_unsequeeze(key.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            value = ckbd_anchor_unsequeeze(value.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            query = ckbd_nonanchor_unsequeeze(query.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention + self.mlp(attention)


class LinearGlobalInterContext(nn.Module):
    """Context module in MLIC++."""

    def __init__(
            self,
            dim=32,
            out_dim=64,
            num_heads=2) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.reprojection = nn.Conv2d(dim, out_dim * 3 // 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim * 2, kernel_size=3, stride=1, padding=1, groups=out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1)
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        B, C, H, W = x1.shape
        queries = self.queries(x1).reshape(B, self.dim, H * W)
        keys = self.keys(x1).reshape(B, self.dim, H * W)
        values = self.values(x1).reshape(B, self.dim, H * W)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(keys[:, i * head_dim: (i + 1) * head_dim, :], dim=2)
            query = F.softmax(queries[:, i * head_dim: (i + 1) * head_dim, :], dim=1)
            value = values[:, i * head_dim: (i + 1) * head_dim, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return self.skip(attention) + self.mlp(attention)
    

class EntropyParameters(nn.Module):
    """Entropy parameters network in MLIC++ with optional domain adapters."""

    def __init__(self, in_dim, out_dim, r=10, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 320, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=1, stride=1, padding=0),
        )
        self.use_adapter = False
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = r

    def set_adapter(self):
        """Construct domain adapters."""

        self.use_adapter = True

        # set LoRA_Adapters
        self.adapter2_A1 = nn.Parameter(torch.randn(self.rank, self.in_dim))
        self.adapter2_A2 = nn.Parameter(torch.randn(self.rank, 320))
        self.adapter2_A3 = nn.Parameter(torch.randn(self.rank, 256))
        self.adapter2_A4 = nn.Parameter(torch.randn(self.rank, 128))
        self.adapter2_B1 = nn.Parameter(torch.zeros(320, self.rank))
        self.adapter2_B2 = nn.Parameter(torch.zeros(256, self.rank))
        self.adapter2_B3 = nn.Parameter(torch.zeros(128, self.rank))
        self.adapter2_B4 = nn.Parameter(torch.zeros(self.out_dim, self.rank))
        nn.init.kaiming_uniform_(self.adapter2_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A4, a=math.sqrt(5))

    def forward(self, params):
        gaussian_params = self.fusion(params)
        return gaussian_params
    
    def forward(self, x):
        if self.use_adapter:
            out = LoRA_forward(self.fusion[0], x, self.adapter2_A1, self.adapter2_B1)
            out = self.fusion[1](out)
            out = LoRA_forward(self.fusion[2], out, self.adapter2_A2, self.adapter2_B2)
            out = self.fusion[3](out)
            out = LoRA_forward(self.fusion[4], out, self.adapter2_A3, self.adapter2_B3)
            out = self.fusion[5](out)
            out = LoRA_forward(self.fusion[6], out, self.adapter2_A4, self.adapter2_B4)
        else:
            out = self.fusion(x)
        return out
    

class LatentResidualPrediction(nn.Module):
    """LRP in MLIC++."""

    def __init__(self, in_dim, out_dim, act=nn.GELU):
        super().__init__()
        diff = abs(out_dim - in_dim)
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, in_dim - diff // 4),
            act(),
            conv3x3(in_dim - diff // 4, in_dim - diff // 2),
            act(),
            conv3x3(in_dim - diff // 2, in_dim - diff * 3 // 4),
            act(),
            conv3x3(in_dim - diff * 3 // 4, out_dim),
        )

    def forward(self, x):
        x = self.lrp_transform(x)
        x = 0.5 * torch.tanh(x)
        return x


class MLICPlusPlus_Adapt(CompressionModel):
    """MLIC++ with optional domain adapters."""

    def __init__(self, **kwargs):
        super().__init__(192)

        config = model_config()

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = M // slice_num

        self.N = N
        self.M = M
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)
        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch)
            for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

    def set_adapter(self):
        """Construct domain adapters."""

        # set Conv_Adapters
        self.g_a.set_adapter()
        self.g_s.set_adapter()

        # set LoRA_Adapters
        for module in self.entropy_parameters_anchor:
            module.set_adapter()
        for module in self.entropy_parameters_nonanchor:
            module.set_adapter()

    def forward(self, x):
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use spatial context, channel context and hyper params)
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def forward_round(self, x):
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z, training=False)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use spatial context, channel context and hyper params)
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def update_resolutions(self, H, W):
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=None)
            else:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=self.local_context[0].attn_mask)

    def compress(self, x):
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
