import torch
import torch.nn as nn
import numpy as np

class HermiteKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HermiteKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        x = torch.tanh(x)

        hermite = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i - 1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()

        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(-1, self.output_dim)
        return y

    def __repr__(self):
        return (f"HermiteKANLayer(input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"degree={self.degree})")

class HermiteKAN(nn.Module):
    def __init__(self, layer_dims, degree=3):
        super(HermiteKAN, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(HermiteKANLayer(layer_dims[i], layer_dims[i + 1], degree))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        layer_str = "\n".join([f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"HermiteKAN(\n  (layers): ModuleList(\n{layer_str}\n  )\n)"

if __name__ == "__main__":
    model = HermiteKAN([2048,1536, 1024, 256, 128,1])
    print(model)

