use game_engine::{Action, GameConfig, GameEngine, GameState};
use game_renderer::{HeadlessRenderer, Renderer, WindowRenderer};
use macroquad::input::{KeyCode, is_key_down};
use macroquad::window::next_frame;
use serde::{Deserialize, Serialize};
use std::net::TcpStream;
use std::time::Instant;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};

pub struct GameSession<T: Renderer> {
    engine: GameEngine,
    renderer: T,
    agent1: Box<dyn Agent>,
    agent2: Box<dyn Agent>,
    step_count: u32,
}

impl GameSession<WindowRenderer> {
    pub async fn run(&mut self) {
        let start_time = Instant::now();
        let mut step_time = Instant::now();

        loop {
            let state = self.engine.get_state();

            let actions = (self.agent1.get_action(state), self.agent2.get_action(state));

            self.engine.step(actions, step_time.elapsed().as_secs_f32());
            step_time = Instant::now();

            let state = self.engine.get_state();
            self.renderer.render(state);

            if state.is_terminal() {
                break;
            }

            next_frame().await;
            self.step_count += 1;
        }
        println!(
            "Session ran for {} steps in {:.2?} seconds",
            self.step_count,
            start_time.elapsed().as_secs_f32()
        );
    }
}

impl GameSession<HeadlessRenderer> {
    pub async fn run(&mut self, delta_time: f32) {
        let start_time = Instant::now();
        loop {
            let state = self.engine.get_state();

            let actions = (self.agent1.get_action(state), self.agent2.get_action(state));

            self.engine.step(actions, delta_time);

            let state = self.engine.get_state();
            self.renderer.render(state);

            if state.is_terminal() {
                break;
            }

            next_frame().await;
            self.step_count += 1;
        }
        println!(
            "Session ran for {} steps in {:.2?} seconds",
            self.step_count,
            start_time.elapsed().as_secs_f32()
        );
    }
}

pub trait Agent {
    fn get_action(&mut self, state: &GameState) -> Action;

    // Optional: for agents that need setup/teardown
    fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn cleanup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

struct HumanAgent {}

impl Agent for HumanAgent {
    fn get_action(&mut self, _state: &GameState) -> Action {
        Action {
            left: is_key_down(KeyCode::Left) | is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::Right) | is_key_down(KeyCode::D),
            up: is_key_down(KeyCode::Up) | is_key_down(KeyCode::W),
            down: is_key_down(KeyCode::Down) | is_key_down(KeyCode::S),
            rotate_left: is_key_down(KeyCode::Q),
            rotate_right: is_key_down(KeyCode::E),
        }
    }
}

struct HeuristicAgent {
    player_id: u8,
}

impl Agent for HeuristicAgent {
    fn get_action(&mut self, state: &GameState) -> Action {
        let mut down = false;
        let mut up = false;
        let mut pos;
        let mut ball_pos = state.get_ball().get_pos();
        if self.player_id == 1 {
            pos = state.get_player1().get_pos();
            pos[0] = 1.0 - pos[0];
            ball_pos[0] = 1.0 - ball_pos[0];
        } else {
            pos = state.get_player2().get_pos();
        }

        if ball_pos[0] > pos[0] {
            if ball_pos[1] > pos[1] {
                down = true;
            } else if ball_pos[1] < pos[1] {
                up = true;
            }
        } else if ball_pos[1] > pos[1] {
            up = true;
        } else if ball_pos[1] < pos[1] {
            down = true;
        }

        Action {
            left: false,
            right: false,
            up,
            down,
            rotate_left: false,
            rotate_right: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PythonRequest {
    state: GameState, // You'll need to implement Serialize for GameState
    player_id: u8,
}

#[derive(Serialize, Deserialize)]
struct PythonResponse {
    action: Action, // You'll need to implement Deserialize for Action
}

pub fn create_python_agent(
    player_id: u8,
    python_url: &str,
) -> Result<Box<dyn Agent>, Box<dyn std::error::Error>> {
    let agent = PythonAgent::new(player_id, python_url);
    match agent {
        Ok(agent) => Ok(Box::new(agent)),
        Err(err) => Err(err),
    }
}

struct PythonAgent {
    player_id: u8,
    websocket: WebSocket<MaybeTlsStream<TcpStream>>,
    connected: bool,
}

impl PythonAgent {
    pub fn new(player_id: u8, python_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (socket, response) = connect(python_url).expect("Can't connect");

        println!("Connected to the server");
        println!("Response HTTP code: {}", response.status());
        println!("Response contains the following headers:");

        Ok(Self {
            player_id,
            websocket: socket,
            connected: true,
        })
    }
}

impl Agent for PythonAgent {
    fn get_action(&mut self, state: &GameState) -> Action {
        if !self.connected {
            return Action::default();
        }

        let request = PythonRequest {
            state: state.clone(),
            player_id: self.player_id,
        };

        let request_json = match serde_json::to_string(&request) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("Failed to serialize request: {e}");
                return Action::default();
            }
        };

        // Send request to Python
        if let Err(e) = self.websocket.send(Message::Text(request_json.into())) {
            eprintln!("Failed to send request: {e}");
            self.connected = false;
            return Action::default();
        }

        match self.websocket.read() {
            Ok(msg @ Message::Text(_)) => match msg.to_text() {
                Ok(text) => match serde_json::from_str::<PythonResponse>(text) {
                    Ok(python_response) => python_response.action,
                    Err(e) => {
                        eprintln!("Failed to parse response: {e}");
                        Action::default()
                    }
                },
                Err(e) => {
                    eprintln!("Failed to parse message: {e}");
                    Action::default()
                }
            },
            _ => {
                eprintln!("Unexpected message type from Python agent");
                Action::default()
            }
        }
    }
}

pub async fn run_interactive_mode() {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);
    let renderer = WindowRenderer::new();
    let agent1 = Box::new(HumanAgent {});
    let agent2 = create_python_agent(2, "ws://localhost:8765").unwrap_or_else(|e| {
        eprintln!("Failed to connect to Python agent, falling back to heuristic: {e}");
        Box::new(HeuristicAgent { player_id: 2 })
    });
    let step_count = 0;
    let mut session = GameSession {
        engine,
        renderer,
        agent1,
        agent2,
        step_count,
    };
    session.run().await;
}

pub async fn run_agent_headless_mode(delta_time: f32) {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);
    let agent1 = create_python_agent(1, "ws://localhost:8765").unwrap_or_else(|e| {
        eprintln!("Failed to connect to Python agent, falling back to heuristic: {e}");
        Box::new(HeuristicAgent { player_id: 1 })
    });
    let agent2 = create_python_agent(2, "ws://localhost:8765").unwrap_or_else(|e| {
        eprintln!("Failed to connect to Python agent, falling back to heuristic: {e}");
        Box::new(HeuristicAgent { player_id: 2 })
    });
    let step_count = 0;
    let mut session = GameSession {
        engine,
        renderer: HeadlessRenderer::new(),
        agent1,
        agent2,
        step_count,
    };
    session.run(delta_time).await;
}
