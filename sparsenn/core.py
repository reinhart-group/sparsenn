import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiTaskMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError('Invalid reduction method')
        self.reduction = reduction

    def forward(self, input, target):
        # Mask out the NaN values in the targets
        mask = ~torch.isnan(target)

        # Compute the mean squared error, ignoring NaN values
        mse = F.mse_loss(input[mask], target[mask], reduction=self.reduction)

        return mse


class L0Linear(nn.Module):
    def __init__(self, in_features, out_features, gamma, zeta, beta, bias=True):
        super(L0Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.log_alpha = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.training:
            gates = self.sample_gates()
        else:
            gates = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            gates = torch.clamp(gates, 0, 1)

        weight = self.weight * gates

        if self.bias is not None:
            return F.linear(input, weight, self.bias)
        else:
            return F.linear(input, weight)

    def sample_gates(self):
        u = torch.rand_like(self.log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        return torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def regularization_term(self):
        return torch.sum(torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta)))

    def count_non_gated_params(self):
        gates = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        gates = torch.clamp(gates, 0, 1)
        non_gated_params = (self.weight * gates).nonzero().size(0)
        return non_gated_params


class MonotonicL0Linear(L0Linear):
    def forward(self, input):
        if self.training:
            gates = self.sample_gates()
        else:
            gates = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            gates = torch.clamp(gates, 0, 1)

        weight = self.weight * gates
        weight = F.relu(weight)  # Apply ReLU to the weights

        if self.bias is not None:
            bias = F.relu(self.bias)  # Apply ReLU to the biases
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight)


class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.relu, gamma=-0.1, zeta=1.1, beta=2 / 3):
        super(RegularizedMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta

        self.layers = nn.ModuleList()
        self.layers.append(self.create_linear_layer(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(self.create_linear_layer(hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(self.create_linear_layer(hidden_dims[-1], output_dim))

    def create_linear_layer(self, in_features, out_features):
        return L0Linear(in_features, out_features, self.gamma, self.zeta, self.beta)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def regularization_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, L0Linear):
                reg_loss += layer.regularization_term()
        return reg_loss


class RegularizedMLPwithSkips(RegularizedMLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.relu, gamma=-0.1, zeta=1.1, beta=2 / 3):
        super().__init__(input_dim, hidden_dims, output_dim, activation=F.relu, gamma=gamma, zeta=zeta, beta=beta)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta

        self.layers = nn.ModuleList()
        self.layers.append(self.create_linear_layer(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(self.create_linear_layer(input_dim + hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(self.create_linear_layer(input_dim + hidden_dims[-1], output_dim))

    def forward(self, x):
        input = x
        x = self.activation(self.layers[0](x))
        for i, layer in enumerate(self.layers[1:-1]):
            x = self.activation(layer(torch.cat([input, x], dim=1)))
        x = self.layers[-1](torch.cat([input, x], dim=1))
        return x


class MonotonicRegularizedMLP(RegularizedMLP):
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=-0.1, zeta=1.1, beta=2 / 3):
        super(MonotonicRegularizedMLP, self).__init__(input_dim, hidden_dims, output_dim, activation=F.relu, gamma=gamma, zeta=zeta, beta=beta)

    def create_linear_layer(self, in_features, out_features):
        return MonotonicL0Linear(in_features, out_features, self.gamma, self.zeta, self.beta)


# TODO: check the following implementation for desired behavior
class ICNN(RegularizedMLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.relu, gamma=-0.1, zeta=1.1, beta=2 / 3):
        super(ICNN, self).__init__(input_dim, hidden_dims, output_dim, activation, gamma, zeta, beta)

        # Additional layers for ICNN
        self.input_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.input_layers.append(self.create_linear_layer(input_dim, hidden_dim))

        self.output_layer_x0 = self.create_linear_layer(input_dim, output_dim)

    def forward(self, x):
        x0 = x

        # First layer
        x = self.activation(self.layers[0](x) + self.input_layers[0](x0))

        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            hidden_weight = F.relu(self.layers[i].linear.weight)
            input_weight = F.relu(self.input_layers[i].linear.weight)
            x = self.activation(F.linear(x, hidden_weight) + F.linear(x0, input_weight) + self.layers[i].linear.bias)

        # Output layer
        output_weight_hidden = F.relu(self.layers[-1].linear.weight)
        output_weight_x0 = F.relu(self.output_layer_x0.linear.weight)
        output = F.linear(x, output_weight_hidden) + F.linear(x0, output_weight_x0) + self.layers[-1].linear.bias
        return output
