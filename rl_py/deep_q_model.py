from typing import final, override
import torch.nn as nn
import torch


@final
class DeepQModel(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int):
        super(DeepQModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Value stream
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),  # Advantage stream
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.network(x)
        value = self.value_stream(base_out)
        advantage = self.advantage_stream(base_out)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
