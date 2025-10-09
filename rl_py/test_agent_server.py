from python_agent_server import PythonAgentServer, BATCH_SIZE
import copy

TEST_STATE = {
    "player1": {
        "pos": [0.0, 0.0],
        "speed": [1.0, 1.0],
    },
    "player2": {
        "pos": [1.0, 1.0],
        "speed": [1.0, 1.0],
    },
    "ball": {
        "pos": [0.5, 0.5],
        "speed": [1.0, 1.0],
    },
    "score": [1, 0],
    "elapsed_time": 60.0,
    "player_id": 1,
    "game_uuid": "id",
}


def test_get_action():
    server = PythonAgentServer(None, False)
    action = server.get_action(copy.deepcopy(TEST_STATE))
    assert isinstance(action, dict)


def test_train():
    server = PythonAgentServer(None, True)
    for _ in range(BATCH_SIZE + 2):
        server.get_action(copy.deepcopy(TEST_STATE))
    server.train()


if __name__ == "__main__":
    test_get_action()
    test_train()
