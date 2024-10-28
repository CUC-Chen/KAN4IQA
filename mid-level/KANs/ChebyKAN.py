import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.input_dim, 1)).expand(-1, -1, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.output_dim)
        return y

    def __repr__(self):
        return f"ChebyKANLayer(input_dim={self.input_dim}, output_dim={self.output_dim}, degree={self.degree})"

class ChebyKAN(nn.Module):
    def __init__(self, layer_dims, degree=3):
        super(ChebyKAN, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(ChebyKANLayer(layer_dims[i], layer_dims[i + 1], degree))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        layer_str = "\n".join([f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"ChebyKAN(\n  (layers): ModuleList(\n{layer_str}\n  )\n)"


