from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        # conv 后的 size 是 n - kernel_size + 1 + 2*padding. 带入 kernel=3, padding=2，
        # 可知 conv 后 siz 不会变
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        # Avg Pooling，可知 h, w 之后成为 H/2, W/2 均减为一半
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            # 每层的输入是 block_expansion * 2 ** i；输出是 block_expansion * 2 **(i + 1)
            # 例外：
            # - 第一层输入是 in_feature
            # - block_expansion * 2 ** i 也不是无限增加的，最大的 size 是 max_features
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        # 输入是 BCHW
        # 输出是一个列表，假设 num_blocks = 2, 则输出是 [x, block1_out, block2_out]
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        # block-expansion, 就对应 conv 最终的 channel 大小，也是每层间缩放的基数；和 Encoder 对应的

        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            # i 最开始是 num_blocsk - 1, 所以 in_filer 一开始是 1 * min(max_feature, block_expansion * 2**num_blocker)
            # 和 Encoder 的最后一层的 channel 是对应的
            # 后续层，要乘以 2；从后面看到，是因为要做残差连接，把 encoder 相应的层输出，以及 decoder 上一层输出 concat 起来输进来；所以乘以 2
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2
             ** (i + 1)))
            # 输出 channel 在变小(因为  h, w 不断扩大；和 Encoder 反过来)
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        # block_expansion 对应 UpBlock 的输出；in_features 对应 encoder 过来的skip-connection
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        # x 是对应 encoder 的各个层的输出（列表），List[0层(原始输入）， 1层，...., N 层]
        # 每层的 shape 是 BCHW 
        # 例子： 
        # - 假设 num_blocks = 2, 则输入是 [original_x, block1_out, block2_out]
        out = x.pop()
        for up_block in self.up_blocks:
            # decoder 从小到大数，
            # - 第 1 层的输入是 [block2_out],
            # - 第 2 层的输入是 cat([decoder_layer1_out, block1_out])
            #      - prev_decoder_out 和 block1_out 的 shape 应该是一模一样的？
            out = up_block(out)
            skip = x.pop()
            # 拼接 skip-connection
            out = torch.cat([out, skip], dim=1)
        # 最后的输出，是 cat([decoder_layer2_out, original_x])
        #   - 两个的 shape, 在 channel 维上不一样；
        # shape 是 batch-size x (original-channel + block_expansion) x H x W.
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    # 沙漏结构
    https://towardsdatascience.com/using-hourglass-networks-to-understand-human-poses-1e40e349fa15
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        # 输入的 shape 是 batch-size x channel x H x W
        # 输出的 shape 是 batch-size x out_filters x H x W
        #   - out_filters size = input-channel + block_expansion
        #   - 注意，它和输入 x 的 shape 不同？并非一个完整的 AutoEncoder?
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    # 这块看代码看不懂；需要一些背景知识才行
    # Anti-Alias 的插值，是通过计算算出一个 conv2d 的 kernel，然后用这个 kernel 来做 conv2d 得到的
    # 关键就是为什么要这么做（比如 hourglass 网络里就直接线性的下采样了；）
    # 为什么该这么做？
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
