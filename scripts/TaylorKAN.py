import torch
import torch.nn as nn

class TaylorLayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, expansion_point=0.0, addbias=True):
        super(TaylorLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias
        self.expansion_point = expansion_point 
        
        # Initialize coefficients for Taylor expansion
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))
    
    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        
        # Shift x by expansion point
        x_shifted = x - self.expansion_point
        
        x_expanded = x_shifted.unsqueeze(1).expand(-1, self.out_dim, -1)
        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)
        
        # Calculate Taylor expansion around the expansion point
        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)
        
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
    
    def __repr__(self):
        return (f"TaylorLayer(input_dim={self.input_dim}, out_dim={self.out_dim}, "
                f"order={self.order}, expansion_point={self.expansion_point}, addbias={self.addbias})")

class TaylorKAN(nn.Module):
    def __init__(self, layers_hidden, order=2, expansion_point=0.0, addbias=True):
        super(TaylorKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(TaylorLayer(in_features, out_features, order, expansion_point, addbias))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        layer_reprs = "\n  ".join([repr(layer) for layer in self.layers])
        return f"TaylorKAN(\n  (layers): ModuleList(\n  {layer_reprs}\n  )\n)"
