import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Callable


class ActLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: Callable[[torch.Tensor], torch.Tensor] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if act == None:
            self.act = None
        else:
            self.act = act

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.act == None:
            output = self.linear(input)
        else:
            output = self.act(self.linear(input))
        return output


class SineLayer(ActLayer):
    def __init__(self, in_features: int, out_features: int, is_first=False, omega_0=30) -> None:
        super().__init__(in_features, out_features, torch.sin)
        self.omega_0 = omega_0

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0,
                                            np.sqrt(6 / in_features) / omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(self.omega_0*input)


class MLP(nn.Module):
    def __init__(self,
                 in_features=2,
                 out_features=3,
                 hidden_layers=3,
                 hidden_features=256,
                 ) -> None:
        super().__init__()
        self.net = []
        self.net.append(
            ActLayer(in_features, hidden_features, act=torch.relu))

        for _ in range(hidden_layers):
            self.net.append(
                ActLayer(hidden_features, hidden_features, act=torch.relu))

        self.net.append(
            ActLayer(hidden_features, out_features, act=torch.tanh))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output


class HashMLP(nn.Module):
    def __init__(self,
                 hash_table_length=[256,256],
                 in_features=2,
                 out_features=3,
                 hidden_layers=2,
                 hidden_features=64) -> None:
        super().__init__()

        self.hash_table = nn.parameter.Parameter(torch.rand(
            (hash_table_length[0], hash_table_length[1], in_features))*1e-4, requires_grad=True)

        self.net = []
        self.net.append(
            ActLayer(in_features, hidden_features, act=torch.relu))

        for _ in range(hidden_layers):
            self.net.append(
                ActLayer(hidden_features, hidden_features, act=torch.relu))

        self.net.append(
            ActLayer(hidden_features, out_features, act=torch.tanh))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        net_in = F.grid_sample(self.hash_table.unsqueeze(0).permute(0, 3, 1, 2),
                               coords.unsqueeze(0), align_corners=True).squeeze(0).permute(1, 2, 0)
        output = self.net(net_in)
        
        return output


class Siren(nn.Module):
    def __init__(self,
                 in_features=2,
                 out_features=3,
                 hidden_layers=3,
                 hidden_features=256,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30) -> None:
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                        is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            self.net.append(SineLayer(
                hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            self.net.append(
                ActLayer(hidden_features, out_features, torch.tanh))
        else:
            self.net.append(SineLayer(
                hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output


class HashINR(nn.Module):
    def __init__(self,
                 hash_table_length=[256,256],
                 in_features=2,
                 out_features=3,
                 hidden_layers=2,
                 hidden_features=64,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30):
        super().__init__()

        self.hash_table = nn.parameter.Parameter(torch.rand(
            (hash_table_length[0], hash_table_length[1], in_features))*1e-4, requires_grad=True)


        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                        is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            self.net.append(SineLayer(
                hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            self.net.append(
                ActLayer(hidden_features, out_features, torch.tanh))
        else:
            self.net.append(SineLayer(
                hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        net_in = F.grid_sample(self.hash_table.unsqueeze(0).permute(0, 3, 1, 2),
                               coords.unsqueeze(0), align_corners=True).squeeze(0).permute(1, 2, 0)
        output = self.net(net_in)

        return output
