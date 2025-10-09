import asyncio
import websockets
import json
import logging
import functools
import argparse

from python_agent_server import PythonAgentServer, default_action

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def handle_agent(
    websocket: websockets.WebSocketServerProtocol, agent: PythonAgentServer
) -> None:
    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                state = request.get("state", {})
                player_id = request.get("player_id", 1)
                state["player_id"] = player_id

                logger.debug(f"Received request for player {player_id}")

                # Get action from neural network
                action = agent.get_action(state)

                response = {"action": action}
                logger.debug(f"Sending response for player {player_id}: {response}")
                await websocket.send(json.dumps(response))

                agent.train()
                agent.update_model()

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                print(e.with_traceback)
                await websocket.send(json.dumps({"action": default_action()}))
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(e.with_traceback)
                await websocket.send(json.dumps({"action": default_action()}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in handle_agent: {e}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="RLPy Training Server")
    _ = parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint file",
    )
    _ = parser.add_argument(
        "--training_mode",
        action="store_true",
        default=False,
        help="Training mode",
    )
    args = parser.parse_args()

    agent = PythonAgentServer(
        checkpoint_path=args.checkpoint_path, training_mode=args.training_mode
    )
    try:
        server = await websockets.serve(
            functools.partial(handle_agent, agent=agent), "localhost", 8765
        )
        logger.info("Python agent server running on ws://localhost:8765")
        await server.wait_closed()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
