import asyncio
import websockets
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PythonAgent:
    def __init__(self) -> None:
        self.model = SimpleNN(10, 6)  # Adjust dimensions as needed
        # Load trained model weights here
        logger.info("PythonAgent initialized")

    def get_action(self, state: Dict[str, Any]) -> Dict[str, bool]:
        try:
            # Convert state to tensor, run through model
            state_tensor = torch.tensor(
                self._state_to_features(state), dtype=torch.float32
            )
            with torch.no_grad():
                output = self.model(state_tensor.unsqueeze(0))  # Add batch dimension
            action_probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(action_probs).item()
            return self._index_to_action(action_idx)
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            return self._get_fallback_action()

    def _state_to_features(self, state: Dict[str, Any]) -> List[float]:
        try:
            # Extract features from game state
            # This is a placeholder - adjust based on actual state structure
            features = []

            # Ball position and velocity
            if "ball" in state:
                ball = state["ball"]
                features.extend(ball.get("pos", [0.5, 0.5]))
                features.extend(ball.get("speed", [0.0, 0.0]))
            else:
                features.extend([0.5, 0.5, 0.0, 0.0])

            # Player positions (assuming we have player1 and player2)
            if "player1" in state:
                p1 = state["player1"]
                features.extend(p1.get("pos", [0.8, 0.5]))
            else:
                features.extend([0.8, 0.5])

            if "player2" in state:
                p2 = state["player2"]
                features.extend(p2.get("pos", [0.2, 0.5]))
            else:
                features.extend([0.2, 0.5])

            # Score
            if "score" in state:
                features.extend(state["score"])
            else:
                features.extend([0, 0])

            # Pad or truncate to expected size
            while len(features) < 10:
                features.append(0.0)
            return features[:10]
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0.0] * 10

    def _index_to_action(self, index: int) -> Dict[str, bool]:
        # Convert model output index to Action struct
        # Define 6 basic actions
        actions = [
            {
                "left": False,
                "right": False,
                "up": True,
                "down": False,
                "rotate_left": False,
                "rotate_right": False,
            },  # Up
            {
                "left": False,
                "right": False,
                "up": False,
                "down": True,
                "rotate_left": False,
                "rotate_right": False,
            },  # Down
            {
                "left": True,
                "right": False,
                "up": False,
                "down": False,
                "rotate_left": False,
                "rotate_right": False,
            },  # Left
            {
                "left": False,
                "right": True,
                "up": False,
                "down": False,
                "rotate_left": False,
                "rotate_right": False,
            },  # Right
            {
                "left": False,
                "right": False,
                "up": False,
                "down": False,
                "rotate_left": True,
                "rotate_right": False,
            },  # Rotate Left
            {
                "left": False,
                "right": False,
                "up": False,
                "down": False,
                "rotate_left": False,
                "rotate_right": True,
            },  # Rotate Right
        ]
        return actions[index % len(actions)]

    def _get_fallback_action(self) -> Dict[str, bool]:
        return {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "rotate_left": False,
            "rotate_right": False,
        }


async def handle_agent(websocket: websockets.WebSocketServerProtocol) -> None:
    agent = PythonAgent()

    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                state = request.get("state", {})
                player_id = request.get("player_id", 1)

                logger.debug(f"Received request for player {player_id}")

                # Get action from neural network
                action = agent.get_action(state)

                response = {"action": action}
                logger.debug(f"Sending response for player {player_id}: {response}")
                await websocket.send(json.dumps(response))

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send(
                    json.dumps({"action": agent._get_fallback_action()})
                )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send(
                    json.dumps({"action": agent._get_fallback_action()})
                )

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in handle_agent: {e}")


async def main() -> None:
    try:
        server = await websockets.serve(handle_agent, "localhost", 8765)
        logger.info("Python agent server running on ws://localhost:8765")
        await server.wait_closed()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
