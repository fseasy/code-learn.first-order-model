from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        # predictor 的输出： batch-size x (input-channel + block_expansion) x H x W
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
        # 在 channel 维度上保持数量和 kp 数一致
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            # 如果是 single, 就 1 个；否则和 kp 数量对应
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            # 注意 out_channels 是 jacobian_maps 数量的 4 倍
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            # 将 jacobian 权重置为 0；bias 初始化也是 4 倍，用的是 1,0,0,1 的值，Why？ 
            # TODO: 要看论文才行了
            # 问题： 这里做置为 0 的操作有用吗？ 这些权重是在构造的时候就初始化好的？还是后面会统一初始化？
            # - search 并看了下代码：
            # - torch 在初始化时，一般默认会调用类自己的 reset_parameters 方法，这就是初始化方法
            #   例如 Conv2d, 这个方法就是做了 kaiming_uniform_
            #   所以，构造函数之后，模块的参数已经初始化了；后面除非显式设置，不需要再初始化了
            #  （不像其他的框架，需要显式再调用 Init 的 op)
            # - 所以这里在构造函数之后再 set, 肯定是有效的，且后续不会被覆盖！
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        # shape = bz x channel x h x w
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        # 在 H, W 维度上先拉平成一维，做 softmax 后再变回来。主要就是想在 HxW 的矩阵上算整体的 softmax
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        return out
