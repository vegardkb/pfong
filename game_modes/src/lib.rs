use game_engine::{Action, Ball, GameConfig, GameEngine, GameState};
use game_renderer::{HeadlessRenderer, Renderer, WindowRenderer};
use macroquad::input::{KeyCode, is_key_down};
use macroquad::window::next_frame;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
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

fn calculate_time_to_collision_x(
    pos: [f32; 2],
    speed: [f32; 2],
    linear_friction: f32,
    x: f32,
) -> f32 {
    let t1 = match (x - pos[0]) / speed[0] {
        t if t > 0.0 => t,
        _ => f32::INFINITY,
    };
    let t2 = match ((2.0 - x) - pos[0]) / speed[0] {
        t if t > 0.0 => t,
        _ => f32::INFINITY,
    };

    let t_no_friction = t1.min(t2);

    if linear_friction < 0.001 {
        return t_no_friction;
    }

    let distance = t_no_friction * speed[0];

    let log_term = 1.0 - linear_friction * distance / speed[0];
    if log_term <= 0.0 {
        f32::INFINITY
    } else {
        -1.0 / linear_friction * log_term.ln()
    }
}

fn calculate_position_at_time(
    pos: [f32; 2],
    speed: [f32; 2],
    linear_friction: f32,
    t: f32,
    x_max: f32,
) -> [f32; 2] {
    let mut x;
    let mut y;
    if linear_friction < 0.001 {
        x = pos[0] + speed[0] * t;
        y = pos[1] + speed[1] * t;
    } else {
        x = pos[0] + speed[0] / linear_friction * (1.0 - (-1.0 * linear_friction * t).exp());
        y = pos[1] + speed[1] / linear_friction * (1.0 - (-1.0 * linear_friction * t).exp());
    }
    while x > 2.0 * x_max || x < 0.0 {
        if x > 2.0 * x_max {
            x -= 2.0 * x_max;
        } else if x < 0.0 {
            x += 2.0 * x_max;
        }
    }
    while y > 2.0 || y < 0.0 {
        if y > 2.0 {
            y -= 2.0;
        } else if y < 0.0 {
            y += 2.0;
        }
    }
    if x > x_max {
        x = 2.0 * x_max - x;
    }
    if y > 1.0 {
        y = 2.0 - y;
    }
    [x, y]
}

fn calculate_angle_and_velocity_at_time(
    angle: f32,
    angular_velocity: f32,
    angular_friction: f32,
    time: f32,
) -> (f32, f32) {
    let mut end_angle = angle
        + angular_velocity * time / angular_friction
            * (1.0 - (-1.0 * angular_friction * time).exp());
    let end_angular_velocity = angular_velocity * (-angular_friction * time).exp();
    if !end_angle.is_finite() {
        end_angle = 0.0;
    }
    while end_angle > 2.0 * PI || end_angle < 0.0 {
        if end_angle > 2.0 * PI {
            end_angle -= 2.0 * PI;
        } else if end_angle < 0.0 {
            end_angle += 2.0 * PI;
        }
    }
    (end_angle, end_angular_velocity)
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
            left_right: if is_key_down(KeyCode::A) {
                -1.0
            } else if is_key_down(KeyCode::D) {
                1.0
            } else {
                0.0
            },
            down_up: if is_key_down(KeyCode::S) {
                1.0
            } else if is_key_down(KeyCode::W) {
                -1.0
            } else {
                0.0
            },
            rotate: if is_key_down(KeyCode::Q) {
                -1.0
            } else if is_key_down(KeyCode::E) {
                1.0
            } else {
                0.0
            },
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
            left_right: rand::random(),
            down_up: rand::random(),
            rotate: rand::random(),
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
        let ball = state.get_ball();
        let mut ball_pos = ball.pos;
        let mut ball_speed = ball.speed;

        let player;
        let x_max;
        if self.player_id == 1 {
            player = state.get_player1();
            x_max = 1.0 - player.min_x;
        } else {
            player = state.get_player2();
            x_max = player.max_x;
        }

        let mut pos = player.pos;
        let mut speed = player.speed;
        let mut angle = player.angle;
        let mut angular_velocity = player.angular_velocity;
        let linear_friction = player.linear_friction;
        let angular_friction = player.angular_friction;

        if self.player_id == 1 {
            pos[0] = 1.0 - pos[0];
            ball_pos[0] = 1.0 - ball_pos[0];
            ball_speed[0] = -ball_speed[0];
            speed[0] = -speed[0];
            angle = -angle;
            angular_velocity = -angular_velocity;
        }

        // Simulate ball and player movement
        let time_to_ball = calculate_time_to_collision_x(ball_pos, ball_speed, 0.0, 0.1);

        let player_pos_forecast = if time_to_ball.is_finite() {
            calculate_position_at_time(pos, speed, linear_friction, time_to_ball, x_max)
        } else {
            pos
        };

        let ball_pos_forecast = if time_to_ball.is_finite() {
            calculate_position_at_time(ball_pos, ball_speed, 0.0, time_to_ball, 1.0)
        } else {
            ball_pos
        };

        let error_x = ball_pos_forecast[0] - player_pos_forecast[0];
        let mut error_y = ball_pos_forecast[1] - player_pos_forecast[1];

        // Select side to aim for
        let target_angle = 0.0;
        let target_angular_vel = if error_y > 0.0 {
            error_y -= player.height / 4.0;
            PI
        } else {
            error_y += player.height / 4.0;
            -PI
        };

        let left_right = (2.0 * error_x) / (player.linear_acc * time_to_ball.powi(2));
        let down_up = (2.0 * error_y) / (player.linear_acc * time_to_ball.powi(2));

        let (angle_forecast, angular_vel_forecast) = calculate_angle_and_velocity_at_time(
            angle,
            angular_velocity,
            angular_friction,
            time_to_ball,
        );

        let mut error_angle = angle_forecast - target_angle;
        while error_angle.abs() > PI / 2.0 {
            if error_angle > PI / 2.0 {
                error_angle -= PI;
            } else if error_angle < -PI / 2.0 {
                error_angle += PI;
            }
        }

        let error_angular_vel = angular_vel_forecast - target_angular_vel;

        let rotate = (2.0 * error_angle) / (player.angular_acc * time_to_ball.powi(2))
            + error_angular_vel / player.angular_acc;

        println!("Time to ball: {}", time_to_ball);
        println!("Errors position: x={}, y={}", error_x, error_y);
        println!(
            "Errors angle: angle={}, angular_vel={}",
            error_angle, error_angular_vel
        );

        Action {
            left_right,
            down_up,
            rotate,
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

pub async fn run_interactive_mode(opponent: String) {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);
    let renderer = WindowRenderer::new();

    let agent1 = create_agent(1, ("Human").to_string());
    let agent2 = create_agent(2, opponent);

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
    player_1: &str,
    player_2: &str,
) {
    for i in 0..num_cycles {
        println!("Starting training cycle {i}");
        for j in 0..num_training {
            println!("Starting training session {j}");
            let config = GameConfig::default();
            let engine = GameEngine::new(config);

            let agent1 = create_agent(1, player_1.to_string());
            let agent2 = create_agent(2, player_2.to_string());

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
                let k = (i * num_evaluation + j) % 2;
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
