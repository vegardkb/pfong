from collections import deque
import torch
from typing import final
import random


class Episode:
    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
    ):
        self.state: torch.Tensor = state
        self.action: int = action
        self.reward: float = reward
        self.next_state: torch.Tensor = next_state


@final
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Episode] = deque(maxlen=capacity)

    def push(
        self,
        episode: Episode,
    ):
        self.buffer.append(episode)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = [episode.state for episode in batch]
        actions = [episode.action for episode in batch]
        rewards = [episode.reward for episode in batch]
        next_states = [episode.next_state for episode in batch]

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)
