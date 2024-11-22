import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -1.5,
            grid_max: float = 1.5,
            num_grids: int = 8,
            denominator: float = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class BSRBF_KANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.SiLU,
            grid_range=[-1.5, 1.5],

    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim

        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(self.output_dim, self.input_dim * (grid_size + spline_order)))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size + spline_order)

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(self.input_dim, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.input_dim,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def forward(self, x):
        device = x.device
        x = self.layernorm(x)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        bs_output = self.b_splines(x).view(x.size(0), -1)

        rbf_output = self.rbf(x).view(x.size(0), -1)

        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        return base_output + bsrbf_output


class BSRBF_KAN(torch.nn.Module):

    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.SiLU,
    ):
        super(BSRBF_KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                BSRBF_KANLayer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    model = BSRBF_KAN([15,26,18,12,1])
    print(model)

