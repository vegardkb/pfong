from typing import final, override
import torch.nn as nn
import torch


@final
class DeepQModel(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int):
        super(DeepQModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Value stream
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),  # Advantage stream
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        value = self.fc_value(base_out)
        advantage = self.fc_advantage(base_out)
        return value + (advantage - advantage.mean())
