from sparsenn.core import L0Linear
import torch
from torch import nn


class MonotonicLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, gamma, zeta, beta):
        super(MonotonicLinearLayer, self).__init__()
        self.sublayer = L0Linear(in_features, out_features, gamma, zeta, beta)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.sublayer.weight, a=0.0, b=1.0)
        nn.init.uniform_(self.sublayer.bias, a=0.0, b=1.0)

    def forward(self, input):
        # force positive weights and biases by always passing through relu
        self.sublayer.weight.data = torch.relu(self.sublayer.weight.data)
        self.sublayer.bias.data = torch.relu(self.sublayer.bias.data)
        return self.sublayer(input)


class PositiveMonotonicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=-0.1, zeta=1.1, beta=2 / 3):
        super(PositiveMonotonicMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = nn.ReLU()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta

        self.layers = nn.ModuleList()
        self.layers.append(MonotonicLinearLayer(input_dim, hidden_dims[0], gamma, zeta, beta))
        for i in range(1, len(hidden_dims)):
            self.layers.append(MonotonicLinearLayer(hidden_dims[i - 1], hidden_dims[i], gamma, zeta, beta))
        self.layers.append(MonotonicLinearLayer(hidden_dims[-1], output_dim, gamma, zeta, beta))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def enforce_nonnegative(self):
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, MonotonicLinearLayer):
                    layer.sublayer.weight.data = torch.relu(layer.sublayer.weight.data)
                    layer.sublayer.bias.data = torch.relu(layer.sublayer.bias.data)
