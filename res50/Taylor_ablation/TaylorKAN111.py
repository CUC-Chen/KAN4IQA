import torch
import torch.nn as nn

class TaylorLayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y

    def __repr__(self):
        return (f"TaylorLayer(input_dim={self.input_dim}, out_dim={self.out_dim}, "
                f"order={self.order}, addbias={self.addbias})")


class TaylorKAN111(nn.Module):
    def __init__(self, layers_hidden, order=5, addbias=True):
        super(TaylorKAN111, self).__init__()
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.relu = nn.ReLU()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(TaylorLayer(in_features, out_features, order, addbias))
            self.batchnorms.append(nn.BatchNorm1d(out_features))

            if in_features != out_features:
                self.projections.append(nn.Linear(in_features, out_features))
            else:
                self.projections.append(None)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x

            x = layer(x)

            x = self.batchnorms[i](x)

            x = self.relu(x)

            if self.projections[i] is not None:
                residual = self.projections[i](residual)

            x = x + residual

        return x

    def __repr__(self):
        layer_reprs = "\n  ".join([repr(layer) for layer in self.layers])
        batchnorm_reprs = "\n  ".join([repr(bn) for bn in self.batchnorms])
        projection_reprs = "\n  ".join([repr(proj) if proj is not None else "None" for proj in self.projections])
        return (f"TaylorKANComplete(\n  (layers): ModuleList(\n  {layer_reprs}\n  ),\n"
                f"  (batchnorms): ModuleList(\n  {batchnorm_reprs}\n  ),\n"
                f"  (projections): ModuleList(\n  {projection_reprs}\n  )\n)")

if __name__ == "__main__":
    model = TaylorKANComplete([15, 26, 18, 12, 1])
    print(model)
