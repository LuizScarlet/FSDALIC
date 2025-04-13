import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from modules.layers.adapter import Conv_Adapter, LoRA_forward
from .base import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.ckbd import *


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


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
        "slice_ch": [8, 8, 8, 8, 16, 16, 32, 32, 96, 96],
        "quant": "ste",
    })

    return config
    

class ResidualBottleneck(nn.Module):
    """ResidualBlock in ELIC with optional domain adapters."""

    def __init__(self, N=192, act=nn.ReLU) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )
        self.use_adapter = False
        self.channel = N

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x):
        out = x + self.branch(x)

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class AttentionBlock(nn.Module):
    """AttentionBlock in ELIC with optional domain adapters."""

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )
        self.use_adapter = False
        self.channel = N

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out


class AnalysisTransformEX(nn.Module):
    """g_a in ELIC with optional domain adapters."""

    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),
            AttentionBlock(M),
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
        

class SynthesisTransformEX(nn.Module):
    """g_s in ELIC with optional domain adapters."""

    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3),
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
    

class HyperAnalysisEX(nn.Module):
    """h_a in ELIC."""

    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            act(),
            conv(N, N),
            act(),
            conv(N, N)
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    

class HyperSynthesisEX(nn.Module):
    """h_s in ELIC."""

    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
    

class LocalContextEX(nn.Module):
    """Spatial context module in ELIC."""

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, local_params):
        local_params = self.fusion(local_params)
        return local_params
    

class ChannelContextEX(nn.Module):
    """Channel context module in ELIC."""

    def __init__(self, in_dim, out_dim, act=nn.ReLU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        channel_params = self.fusion(channel_params)
        return channel_params
    

class EntropyParametersEX(nn.Module):
    """Entropy parameters network in ELIC with optional domain adapters."""

    def __init__(self, in_dim, out_dim, act=nn.GELU, r=10) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
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
        self.adapter2_A2 = nn.Parameter(torch.randn(self.rank, self.out_dim * 5 // 3))
        self.adapter2_A3 = nn.Parameter(torch.randn(self.rank, self.out_dim * 4 // 3))
        self.adapter2_B1 = nn.Parameter(torch.zeros(self.out_dim * 5 // 3, self.rank))
        self.adapter2_B2 = nn.Parameter(torch.zeros(self.out_dim * 4 // 3, self.rank))
        self.adapter2_B3 = nn.Parameter(torch.zeros(self.out_dim, self.rank))
        nn.init.kaiming_uniform_(self.adapter2_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A3, a=math.sqrt(5))

    def forward(self, x):
        if self.use_adapter:
            out = LoRA_forward(self.fusion[0], x, self.adapter2_A1, self.adapter2_B1)
            out = self.fusion[1](out)
            out = LoRA_forward(self.fusion[2], out, self.adapter2_A2, self.adapter2_B2)
            out = self.fusion[3](out)
            out = LoRA_forward(self.fusion[4], out, self.adapter2_A3, self.adapter2_B3)
        else:
            out = self.fusion(x)
        return out
    

class ELIC_Adapt(CompressionModel):
    """ELIC with optional domain adapters."""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__()
        config = model_config()
        self.slice_num = config.slice_num
        self.slice_ch = config.slice_ch

        self.g_a = AnalysisTransformEX(N, M)
        self.g_s = SynthesisTransformEX(N, M)
        self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)

        self.local_context = nn.ModuleList(
            LocalContextEX(in_dim=self.slice_ch[i], out_dim=self.slice_ch[i] * 2)
            for i in range(len(self.slice_ch))
        )
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(self.slice_ch[:i]), out_dim=self.slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(self.slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + self.slice_ch[i] * 2, out_dim=self.slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=self.slice_ch[i] * 2, act=nn.ReLU)
            for i in range(self.slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + self.slice_ch[i] * 4, out_dim=self.slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2 + self.slice_ch[i] * 2, out_dim=self.slice_ch[i] * 2, act=nn.ReLU)
            for i in range(self.slice_num)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

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
        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        hyper_params = self.h_s(z_hat)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:

                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor
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
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset
        hyper_params = self.h_s(z_hat)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:

                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor

                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }
    
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
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
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
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
        hyper_params = self.h_s(z_hat)

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
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated