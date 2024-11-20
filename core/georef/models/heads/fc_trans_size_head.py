import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init

from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.conv_module import ConvModule


class FC_TransSizeHead(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        feat_dim=256,
        num_layers=2,
        norm="GN",
        num_gn_groups=32,
        act="gelu",
        num_classes=1,
        kernel_size=1,
        num_points=1,
        per_rot_sup=False,
        norm_input=False,
        dropout=False,
        point_bias=True,
        **args,
    ):
        super(FC_TransSizeHead, self).__init__()
        self.per_rot_sup = per_rot_sup
        self.head_t = RotHead(
            in_dim-3,
            feat_dim,
            num_layers,
            norm,
            num_gn_groups,
            act,
            num_classes,
            kernel_size,
            num_points,
            norm_input,
            dropout,
            point_bias,
        )
        self.head_s = RotHead(
            in_dim,
            feat_dim,
            num_layers,
            norm,
            num_gn_groups,
            act,
            num_classes,
            kernel_size,
            num_points,
            norm_input,
            dropout,
            point_bias,
        )

    def forward(self, x_t, x_s):
        t, feat_t = self.head_t(x_t)
        s, feat_s = self.head_s(x_s)
        return t, s


class RotHead(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        feat_dim=256,
        num_layers=2,
        norm="none",
        num_gn_groups=32,
        act="leaky_relu",
        num_classes=1,
        kernel_size=1,
        num_points=1,
        norm_input=False,
        dropout=False,
        point_bias=True,
    ):
        super().__init__()
        self.norm = get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
        self.act_func = act_func = get_nn_act_func(act)
        self.num_classes = num_classes

        self.layers = nn.ModuleList()

        if norm_input:
            self.layers.append(nn.BatchNorm1d(in_dim))
        for _i in range(num_layers):
            _in_dim = in_dim if _i == 0 else feat_dim
            self.layers.append(nn.Conv1d(_in_dim, feat_dim, kernel_size))
            self.layers.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
            self.layers.append(act_func)
            if dropout:
                self.layers.append(nn.Dropout(p=0.2))

        self.neck = nn.ModuleList()
        self.neck.append(nn.Conv1d(feat_dim, 3 * num_classes, 1))

        self.conv_p = nn.Conv1d(num_points, 1, 1, bias=point_bias)

        # init ------------------------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

    def forward(self, x):
        for _layer in self.layers:
            x = _layer(x)

        for _layer in self.neck:
            x = _layer(x)

        feat = x.clone()
        x = x.permute(0, 2, 1)
        x = self.conv_p(x)

        x = x.squeeze(1)
        x = x.contiguous()

        return x, feat


if __name__ == "__main__":
    points = torch.rand(8, 1088, 1024 + 32)  # bs x feature x num_p
    rot_head = FC_TransSizeHead(in_dim=1088, num_points=1024 + 32)
    rot, feat = rot_head(points)
    print(rot.shape)
    print(feat.shape)
