import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SplineLinear(nn.Module):
    def __init__(self, input_dim, output_dim, init_scale=0.1):
        super(SplineLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.trunc_normal_(self.weight, mean=0, std=init_scale)

    def forward(self, x):
        return torch.matmul(x, self.weight.t())

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0, num_grids=8, denominator=None):
        super(RadialBasisFunction, self).__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(self.grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_min=-2.0, grid_max=2.0, num_grids=8, 
                 use_base_update=True, use_layernorm=True, base_activation=F.silu, init_scale=0.1):
        super(FastKANLayer, self).__init__()
        self.use_base_update = use_base_update
        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm else None
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, init_scale)
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.layernorm:
            x = self.layernorm(x)
        spline_basis = self.rbf(x)
        y = self.spline_linear(spline_basis.view(spline_basis.shape[0], -1))
        if self.use_base_update:
            y += self.base_linear(self.base_activation(x))
        return y

class FastKAN(nn.Module):
    def __init__(self, layers_hidden: List[int], grid_min=-2.0, grid_max=2.0, num_grids=8, 
                 use_base_update=True, base_activation=F.silu, init_scale=0.1):
        super(FastKAN, self).__init__()
        self.layers = nn.ModuleList([FastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, 
                                                  use_base_update, base_activation=base_activation, init_scale=init_scale)
                                     for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionWithFastKANTransform(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, head_dim, num_heads, gating=True):
        super(AttentionWithFastKANTransform, self).__init__()
        self.num_heads = num_heads
        total_dim = head_dim * num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        if gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        self.norm = head_dim**-0.5

    def forward(self, q, k, v, bias=None):
        wq = self.linear_q(q).view(q.shape[0], -1, self.num_heads, -1) * self.norm
        wk = self.linear_k(k).view(k.shape[0], -1, self.num_heads, -1)
        att = torch.softmax((wq * wk).sum(-1), dim=-2)
        if bias is not None:
            att += bias[..., None]
        wv = self.linear_v(v).view(v.shape[0], -1, self.num_heads, -1)
        o = (att[..., None] * wv).sum(-3).view(q.shape[0], -1)
        if self.gating:
            g = torch.sigmoid(self.linear_g(q))
            o *= g
        return self.linear_o(o)
