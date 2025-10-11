from deep_q_model import DeepQModel
from replay_buffer import ReplayBuffer, Episode
from metric_server import MetricServer

from typing import final, Any
import logging
import torch
import torch.nn.functional as F
import random
import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 18
N_ACTIONS = 27
MEMORY_SIZE = 50000
BATCH_SIZE = 512
UPDATE_INTERVAL = 1500
LR = 0.0001
SAVE_INTERVAL = 100000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99998
METRIC_BUFFER_SIZE = 100


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
        action_counts_dict: dict[str, int] = {}
        self.metric_buffers = {
            "loss": np.zeros(METRIC_BUFFER_SIZE),
            "reward": np.zeros(METRIC_BUFFER_SIZE),
            "epsilon": np.zeros(METRIC_BUFFER_SIZE),
            "q_value": np.zeros(METRIC_BUFFER_SIZE),
            "q_std": np.zeros(METRIC_BUFFER_SIZE),
            "action_counts": action_counts_dict,
        }

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
        state_tensor = torch.tensor(
            state_to_features(state, player_id), dtype=torch.float32
        )

        # Select action
        with torch.no_grad():
            output = self.policy_net(state_tensor.unsqueeze(0).to(self.device))

        action_probs = torch.softmax(output, dim=1)
        action_idx = torch.argmax(action_probs).item()
        policy_action, policy_action_str = index_to_action(action_idx, player_id)

        if self.train_mode and not evaluation_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, N_ACTIONS - 1)
            action, _ = index_to_action(action_idx, player_id)
        else:
            action = policy_action

        # Save episode and update training parameters
        if self.train_mode and not evaluation_mode:
            reward = calculate_reward(state, player_id)

            self.metric_buffers["epsilon"][self.steps % METRIC_BUFFER_SIZE] = (
                self.epsilon
            )
            self.metric_buffers["reward"][self.steps % METRIC_BUFFER_SIZE] = reward
            self.metric_buffers["q_value"][self.steps % METRIC_BUFFER_SIZE] = (
                output.max().item()
            )
            if policy_action_str not in self.metric_buffers["action_counts"]:
                self.metric_buffers["action_counts"][policy_action_str] = 0
            self.metric_buffers["action_counts"][policy_action_str] += 1

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
                "reward": reward,
            }

            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            self.steps += 1

        return action

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

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1]
            next_q_values = (
                self.target_net(next_state_batch)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = reward_batch + GAMMA * next_q_values
            target_q_values = torch.clamp(target_q_values, -20.0, 20.0)

        loss = F.mse_loss(q_values, target_q_values)

        self.metric_buffers["loss"][self.steps % METRIC_BUFFER_SIZE] = loss.item()
        if self.steps % METRIC_BUFFER_SIZE == 0 and self.steps > 0:
            self.metric_server.record_metrics(
                {
                    "loss": self.metric_buffers["loss"].mean(),
                    "epsilon": self.metric_buffers["epsilon"].mean(),
                    "reward": self.metric_buffers["reward"].mean(),
                    "q_value": self.metric_buffers["q_value"].mean(),
                    "q_std": self.metric_buffers["q_value"].std(),
                    "action_rate": {
                        k: float(v) / METRIC_BUFFER_SIZE
                        for k, v in self.metric_buffers["action_counts"].items()
                    },
                },
                self.steps,
            )
            for k, v in self.metric_buffers["action_counts"].items():
                self.metric_buffers["action_counts"][k] = 0
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
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
    score_factor = 10.0
    ball_proximity_factor = 1.0
    ball_hit_reward = 2.0

    opponent_id = 3 - player_id  # More reliable: 2 if player_id=1, 1 if player_id=2

    reward: float = 0.0

    # Scoring rewards (sparse but important)
    if state["point_scored"] == player_id:
        reward += score_factor
    elif state["point_scored"] == opponent_id:
        reward -= score_factor

    # Proximity to ball (dense reward)
    # ball_pos = np.array(state["ball"]["pos"])
    # player_pos = np.array(state[f"player{player_id}"]["pos"])
    # distance_to_ball = np.linalg.norm(ball_pos - player_pos)
    # reward += ball_proximity_factor / (1.0 + distance_to_ball)  # Closer = better

    # Reward for hitting the ball
    # if state["ball_hit"] == player_id:
    #    reward += ball_hit_reward

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
                features.append(np.sin(angle))
                features.append(np.cos(angle))

        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        print(e.__traceback__)
        return [0.0] * N_FEATURES


def index_to_action(index: int, player_id: int) -> tuple[dict[str, bool], str]:
    lr = ["left", "none", "right"] if player_id == 1 else ["right", "none", "left"]
    ud = ["up", "none", "down"]
    rot = (
        ["rotate_left", "none", "rotate_right"]
        if player_id == 1
        else ["rotate_right", "none", "rotate_left"]
    )

    action_strs: list[str] = []
    for horizontal in lr:
        for vertical in ud:
            for rotation in rot:
                action_strs.append(f"{horizontal}_{vertical}_{rotation}")

    action = default_action()

    action_str = action_strs[index]

    for direction in action.keys():
        action[direction] = direction in action_str

    return action, action_str


def default_action() -> dict[str, bool]:
    return {
        "left": False,
        "right": False,
        "up": False,
        "down": False,
        "rotate_left": False,
        "rotate_right": False,
    }
