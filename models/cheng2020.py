import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.ans import BufferedRansEncoder, RansDecoder
from modules.layers.adapter import Conv_Adapter, LoRA_forward
from compressai.models import Cheng2020Attention
from compressai.layers import conv3x3, subpel_conv3x3, GDN


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlock(nn.Module):
    """ResidualBlock in Cheng2020 with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
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

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class ResidualBlockWithStride(nn.Module):
    """ResidualBlockWithStride in Cheng2020 with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
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

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)
        
        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class ResidualBlockUpsample(nn.Module):
    """ResidualBlockUpsample in Cheng2020 with optional domain adapters."""

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.use_adapter = False
        self.channel = out_ch

    def set_adapter(self):
        self.use_adapter = True
        self.adapter1 = Conv_Adapter(self.channel)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class AttentionBlock(nn.Module):
    """AttentionBlock in Cheng2020 with optional domain adapters."""

    def __init__(self, N):
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

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity

        if self.use_adapter:
            out = self.adapter1(out)

        return out
    

class Cheng2020Attention_Adapt(Cheng2020Attention):
    """Cheng2020 with optional domain adapters."""

    def __init__(self, N=192, r=10, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )
        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
        )
        self.use_adapter = False
        self.channel = N
        self.rank = r

    def set_adapter(self):
        """Construct domain adapters."""

        self.use_adapter = True

        # set Conv_Adapters
        for module in self.g_a:
            if hasattr(module, "set_adapter"):
                module.set_adapter()
        for module in self.g_s:
            if hasattr(module, "set_adapter"):
                module.set_adapter()

        # set LoRA_Adapters
        self.adapter2_A1 = nn.Parameter(torch.randn(self.rank, self.channel * 12 // 3))
        self.adapter2_A2 = nn.Parameter(torch.randn(self.rank, self.channel * 10 // 3))
        self.adapter2_A3 = nn.Parameter(torch.randn(self.rank, self.channel * 8 // 3))
        self.adapter2_B1 = nn.Parameter(torch.zeros(self.channel * 10 // 3, self.rank))
        self.adapter2_B2 = nn.Parameter(torch.zeros(self.channel * 8 // 3, self.rank))
        self.adapter2_B3 = nn.Parameter(torch.zeros(self.channel * 6 // 3, self.rank))
        nn.init.kaiming_uniform_(self.adapter2_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter2_A3, a=math.sqrt(5))
    
    def entropy_parameters_forward(self, x):
        if self.use_adapter:
            out = LoRA_forward(self.entropy_parameters[0], x, self.adapter2_A1, self.adapter2_B1)
            out = self.entropy_parameters[1](out)
            out = LoRA_forward(self.entropy_parameters[2], out, self.adapter2_A2, self.adapter2_B2)
            out = self.entropy_parameters[3](out)
            out = LoRA_forward(self.entropy_parameters[4], out, self.adapter2_A3, self.adapter2_B3)
        else:
            out = self.entropy_parameters(x)
        return out

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters_forward(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward_round(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=False)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters_forward(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat, training=False)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters_forward(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters_forward(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv