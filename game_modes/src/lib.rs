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

pub struct SessionMetadata {
    pub evaluation_mode: bool,
    pub player1_id: String,
    pub player2_id: String,
}

impl GameSession<WindowRenderer> {
    pub async fn run(&mut self, evaluation_mode: bool) {
        let start_time = Instant::now();
        let mut step_time = Instant::now();

        let metadata = SessionMetadata {
            evaluation_mode,
            player1_id: self.agent1.get_name(),
            player2_id: self.agent2.get_name(),
        };

        loop {
            let state = self.engine.get_state();

            let actions = (
                self.agent1.get_action(state, &metadata),
                self.agent2.get_action(state, &metadata),
            );

            self.engine.step(actions, step_time.elapsed().as_secs_f32());
            step_time = Instant::now();

            let state = self.engine.get_state();
            self.renderer.render(state);

            if state.is_terminal() {
                let _ = (
                    self.agent1.get_action(state, &metadata),
                    self.agent2.get_action(state, &metadata),
                );
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
    pub async fn run(&mut self, delta_time: f32, evaluation_mode: bool) {
        let start_time = Instant::now();

        let metadata = SessionMetadata {
            evaluation_mode,
            player1_id: self.agent1.get_name(),
            player2_id: self.agent2.get_name(),
        };

        loop {
            let state = self.engine.get_state();

            let actions = (
                self.agent1.get_action(state, &metadata),
                self.agent2.get_action(state, &metadata),
            );

            self.engine.step(actions, delta_time);

            let state = self.engine.get_state();
            self.renderer.render(state);

            if state.is_terminal() {
                let _ = (
                    self.agent1.get_action(state, &metadata),
                    self.agent2.get_action(state, &metadata),
                );
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
    fn get_action(&mut self, state: &GameState, metadata: &SessionMetadata) -> Action;

    fn get_name(&self) -> String;

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
    fn get_action(&mut self, _state: &GameState, _metadata: &SessionMetadata) -> Action {
        Action {
            left: is_key_down(KeyCode::Left) | is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::Right) | is_key_down(KeyCode::D),
            up: is_key_down(KeyCode::Up) | is_key_down(KeyCode::W),
            down: is_key_down(KeyCode::Down) | is_key_down(KeyCode::S),
            rotate_left: is_key_down(KeyCode::Q),
            rotate_right: is_key_down(KeyCode::E),
        }
    }

    fn get_name(&self) -> String {
        "Human".to_string()
    }
}

struct RandomAgent {}

impl Agent for RandomAgent {
    fn get_action(&mut self, _state: &GameState, _metadata: &SessionMetadata) -> Action {
        Action {
            left: rand::random(),
            right: rand::random(),
            up: rand::random(),
            down: rand::random(),
            rotate_left: rand::random(),
            rotate_right: rand::random(),
        }
    }

    fn get_name(&self) -> String {
        "Random".to_string()
    }
}

struct HeuristicAgent {
    player_id: u8,
}

impl Agent for HeuristicAgent {
    fn get_action(&mut self, state: &GameState, _metadata: &SessionMetadata) -> Action {
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

    fn get_name(&self) -> String {
        "Heuristic".to_string()
    }
}

#[derive(Serialize, Deserialize)]
struct PythonRequest {
    state: GameState,
    player_id: u8,
    evaluation_mode: bool,
    opponent_id: String,
    game_terminated: bool,
}

#[derive(Serialize, Deserialize)]
struct PythonResponse {
    action: Action,
}

pub fn create_agent(player_id: u8, agent_name: String) -> Box<dyn Agent> {
    match agent_name.as_str() {
        "Heuristic" => Box::new(HeuristicAgent { player_id }),
        "Random" => Box::new(RandomAgent {}),
        "Human" => Box::new(HumanAgent {}),
        "Python" => {
            let agent = PythonAgent::new(player_id, "ws://localhost:8765");
            match agent {
                Ok(agent) => Box::new(agent),
                Err(err) => {
                    eprintln!(
                        "Failed to connect to Python agent, falling back to heuristic: {err}"
                    );
                    Box::new(HeuristicAgent { player_id })
                }
            }
        }
        _ => panic!("Unknown agent name"),
    }
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
        let (socket, _) = connect(python_url).expect("Can't connect");

        Ok(Self {
            player_id,
            websocket: socket,
            connected: true,
        })
    }
}

impl Agent for PythonAgent {
    fn get_action(&mut self, state: &GameState, metadata: &SessionMetadata) -> Action {
        if !self.connected {
            return Action::default();
        }

        let opponent_id = if self.player_id == 1 {
            metadata.player2_id.clone()
        } else {
            metadata.player1_id.clone()
        };

        let request = PythonRequest {
            state: state.clone(),
            player_id: self.player_id,
            opponent_id,
            evaluation_mode: metadata.evaluation_mode,
            game_terminated: state.is_terminal(),
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

    fn get_name(&self) -> String {
        "Python".to_string()
    }
}

pub async fn run_interactive_mode() {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);
    let renderer = WindowRenderer::new();

    let agent1 = create_agent(1, ("Human").to_string());
    let agent2 = create_agent(2, ("Python").to_string());

    let step_count = 0;
    let mut session = GameSession {
        engine,
        renderer,
        agent1,
        agent2,
        step_count,
    };
    let evaluation_mode = true;
    session.run(evaluation_mode).await;
}

pub async fn run_agent_headless_mode(delta_time: f32, num_games: u32) {
    for i in 0..num_games {
        let config = GameConfig::default();
        let engine = GameEngine::new(config);

        let agent1 = create_agent(1, ("Python").to_string());
        let agent2 = create_agent(2, ("Python").to_string());

        let step_count = 0;
        let mut session = GameSession {
            engine,
            renderer: HeadlessRenderer::new(),
            agent1,
            agent2,
            step_count,
        };
        let evaluation_mode = false;
        session.run(delta_time, evaluation_mode).await;
        println!("Finished session {i}");
    }
}

pub async fn run_agent_headless_training_mode(
    delta_time: f32,
    num_cycles: u32,
    num_training: u32,
    num_evaluation: u32,
    opponents: &Vec<String>,
) {
    for i in 0..num_cycles {
        println!("Starting training cycle {i}");
        for j in 0..num_training {
            println!("Starting training session {j}");
            let config = GameConfig::default();
            let engine = GameEngine::new(config);

            let agent1 = create_agent(1, ("Python").to_string());
            let agent2 = create_agent(2, ("Python").to_string());

            let step_count = 0;
            let mut session = GameSession {
                engine,
                renderer: HeadlessRenderer::new(),
                agent1,
                agent2,
                step_count,
            };
            let evaluation_mode = false;
            session.run(delta_time, evaluation_mode).await;
        }
        for opponent in opponents {
            println!("Starting evaluation session against {opponent}");
            for j in 0..num_evaluation {
                println!("Starting evaluation session {j}");
                for k in 0..2 {
                    let agent1;
                    let agent2;
                    if k == 0 {
                        agent1 = create_agent(1, ("Python").to_string());
                        agent2 = create_agent(2, (opponent).to_string());
                    } else {
                        agent1 = create_agent(1, (opponent).to_string());
                        agent2 = create_agent(2, ("Python").to_string());
                    }

                    let config = GameConfig::default();
                    let engine = GameEngine::new(config);
                    let step_count = 0;
                    let mut session = GameSession {
                        engine,
                        renderer: HeadlessRenderer::new(),
                        agent1,
                        agent2,
                        step_count,
                    };
                    let evaluation_mode = true;
                    session.run(delta_time, evaluation_mode).await;
                }
            }
        }
    }
}
