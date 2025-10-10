from deep_q_model import DeepQModel
from replay_buffer import ReplayBuffer, Episode
from metric_server import MetricServer

from typing import final, Any
import logging
import torch
import torch.nn.functional as F
import math
import random
import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 16
N_ACTIONS = 7
MEMORY_SIZE = 100000
BATCH_SIZE = 256
UPDATE_INTERVAL = 1000
LR = 0.0005
SAVE_INTERVAL = 100000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999995
LOSS_BUFFER_SIZE = 1000


@final
class PythonAgentServer:
    def __init__(self, checkpoint_path: str, training_mode: bool) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Device: {self.device}")

        # Initialize the networks
        self.policy_net = DeepQModel(N_FEATURES, N_ACTIONS).to(self.device)
        if checkpoint_path is not None:
            self.policy_net.load_state_dict(torch.load(checkpoint_path))

        self.target_net = DeepQModel(N_FEATURES, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)

        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.last_state_dict = dict()
        self.steps = 0
        self.train_mode = training_mode

        self.active_evaluation_games = {}
        self.loss_buffer = np.zeros(LOSS_BUFFER_SIZE)

        self.epsilon = EPSILON_START

        self.metric_server = MetricServer(port=5001)

        logger.info("PythonAgent initialized")

    def get_action(self, state: dict[str, Any]) -> dict[str, bool]:
        game_uuid = state.pop("game_uuid")
        player_id = state["player_id"]
        evaluation_mode = state["evaluation_mode"]
        game_terminated = state["game_terminated"]
        game_uuid = game_uuid + str(player_id)

        # Detect game start
        if game_uuid not in self.active_evaluation_games and evaluation_mode:
            self.active_evaluation_games[game_uuid] = {
                "opponent_id": state.get("opponent_id", "unknown"),
                "player_id": player_id,
                "start_step": self.steps,
                "last_score": state["score"].copy(),
            }

        # Detect game end (score changed to terminal state)
        if game_uuid in self.active_evaluation_games and game_terminated:
            game_info = self.active_evaluation_games[game_uuid]
            current_score = state["score"]
            my_score = current_score[player_id - 1]
            opponent_score = current_score[2 - player_id]
            won = my_score > opponent_score

            self.metric_server.record_result(
                opponent_id=game_info["opponent_id"],
                player_id=game_info["player_id"],
                score=current_score,
                won=won,
                training_step=game_info["start_step"],
            )
            del self.active_evaluation_games[game_uuid]

        # Process reward and features
        reward = calculate_reward(state, player_id)
        state_tensor = torch.tensor(
            state_to_features(state, player_id), dtype=torch.float32
        )

        # Select action
        if self.train_mode and not evaluation_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                output = self.policy_net(state_tensor.unsqueeze(0).to(self.device))

            action_probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(action_probs).item()

        # Save episode and update training parameters
        if self.train_mode and not evaluation_mode:
            if game_uuid in self.last_state_dict:
                episode = Episode(
                    self.last_state_dict[game_uuid]["state"],
                    self.last_state_dict[game_uuid]["action"],
                    reward,
                    state_tensor,
                )
                self.replay_buffer.push(episode)

            self.last_state_dict[game_uuid] = {
                "state": state_tensor,
                "action": action_idx,
            }

            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            self.steps += 1

        return index_to_action(action_idx, player_id)

    def train(self):
        if not self.train_mode:
            return

        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states = self.replay_buffer.sample(BATCH_SIZE)

        state_batch = torch.stack(states, dim=0).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.stack(next_states, dim=0).to(self.device)

        q_values = (
            self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        )
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + GAMMA * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.loss_buffer[self.steps % LOSS_BUFFER_SIZE] = loss.item()
        if self.steps % LOSS_BUFFER_SIZE == 0 and self.steps > 0:
            self.metric_server.record_metrics(
                {"loss": self.loss_buffer.mean()}, self.steps
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % SAVE_INTERVAL == 0:
            torch.save(
                self.policy_net.state_dict(),
                f"model_checkpoints/model_{self.steps}.pth",
            )

    def update_model(self):
        if not self.train_mode:
            return
        if self.steps % UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def calculate_reward(state: dict[str, Any], player_id: int) -> float:
    reward = 0.0
    if player_id == 1:
        reward += 10.0 * state["score"][0]
        reward -= 10.0 * state["score"][1]
        reward += 0.1 * state["ball"]["speed"][0]
    else:
        reward -= 10.0 * state["score"][0]
        reward += 10.0 * state["score"][1]
        reward -= 0.1 * state["ball"]["speed"][0]

    reward -= 0.01 * state["elapsed_time"]

    return reward


def state_to_features(state: dict[str, Any], player_id: int) -> list[float]:
    entity_order = (
        ["ball", "player1", "player2"]
        if player_id == 1
        else ["ball", "player2", "player1"]
    )
    flip_x_axes = player_id == 2

    try:
        features = []

        for entity in entity_order:
            entity_dict = state[entity]
            pos = entity_dict["pos"]
            speed = entity_dict["speed"]
            if flip_x_axes:
                pos = [1.0 - pos[0], pos[1]]
                speed = [-speed[0], speed[1]]
            features.extend(pos)
            features.extend(speed)

            if "angular_velocity" in entity_dict:
                rotation = entity_dict["angular_velocity"]
                angle = entity_dict["angle"]
                if flip_x_axes:
                    rotation = -rotation
                    angle = -angle
                features.append(rotation)
                features.append(angle % (2 * math.pi))

        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        print(e.__traceback__)
        return [0.0] * N_FEATURES


def index_to_action(index: int, player_id: int) -> dict[str, bool]:
    directions = ["left", "right", "up", "down", "rotate_left", "rotate_right"]
    directions_alt = ["right", "left", "up", "down", "rotate_right", "rotate_left"]

    action = default_action()

    if index > 0:
        if player_id == 1:
            action[directions[index - 1]] = True
        else:
            action[directions_alt[index - 1]] = True

    return action


def default_action() -> dict[str, bool]:
    return {
        "left": False,
        "right": False,
        "up": False,
        "down": False,
        "rotate_left": False,
        "rotate_right": False,
    }
