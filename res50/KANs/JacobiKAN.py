import torch
import torch.nn as nn
import numpy as np

class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.a = a
        self.b = b

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        x = torch.tanh(x)
        jacobi = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2) / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.output_dim)
        return y

    def __repr__(self):
        return f"JacobiKANLayer(input_dim={self.input_dim}, output_dim={self.output_dim}, degree={self.degree}, a={self.a}, b={self.b})"

class JacobiKAN(nn.Module):
    def __init__(self, layer_dims, degree=3, a=0.9, b=0.9):
        super(JacobiKAN, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(JacobiKANLayer(layer_dims[i], layer_dims[i + 1], degree, a, b))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        layer_str = "\n".join([f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"JacobiKAN(\n  (layers): ModuleList(\n{layer_str}\n  )\n)"

if __name__ == "__main__":
    model = JacobiKAN([2048,1536, 1024, 256, 128,1])
    print(model)
