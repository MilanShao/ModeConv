from typing import Optional

import numpy as np
import scipy.ndimage
from scipy import linalg as LA
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (
    add_self_loops,
    get_laplacian,
    remove_self_loops,
)


class ModeConvFast(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, num_sensors,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.num_sensors = num_sensors
        self.lins_r = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])
        self.lins_i = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins_i:
            lin.reset_parameters()
        for lin in self.lins_r:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor,
                weight: OptTensor = None, sim=None):

        weight = weight * sim[..., None]
        Tx_1 = self.propagate(edge_index, x=x, weight=weight, size=None)
        Tx_1_c = torch.view_as_real(Tx_1)
        out_r = self.lins_r[0](Tx_1_c[..., 0]).view((-1, self.out_channels))
        out_i = self.lins_i[0](Tx_1_c[..., 1]).view((-1, self.out_channels))
        out = out_r - out_i

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, weight):
        r = weight.reshape(-1, 1, 2)[..., 0] * x_j
        i = weight.reshape(-1, 1, 2)[..., 1] * x_j
        return torch.complex(r, i)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')
