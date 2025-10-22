use game_engine::{Action, GameConfig, GameEngine, GameState};
use game_renderer::{HeadlessRenderer, Renderer, WindowRenderer};
use macroquad::audio::{PlaySoundParams, load_sound, play_sound};
use macroquad::input::{KeyCode, is_key_down};
use macroquad::prelude::set_pc_assets_folder;
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

fn calculate_state_at_time(
    pos: [f32; 2],
    speed: [f32; 2],
    linear_friction: f32,
    t: f32,
    x_max: f32,
) -> ([f32; 2], [f32; 2]) {
    let mut x;
    let mut y;
    let mut vx;
    let mut vy;
    if linear_friction < 0.001 {
        x = pos[0] + speed[0] * t;
        y = pos[1] + speed[1] * t;
        vx = speed[0];
        vy = speed[1];
    } else {
        x = pos[0] + speed[0] / linear_friction * (1.0 - (-1.0 * linear_friction * t).exp());
        y = pos[1] + speed[1] / linear_friction * (1.0 - (-1.0 * linear_friction * t).exp());
        vx = speed[0] * (-1.0 * linear_friction * t).exp();
        vy = speed[1] * (-1.0 * linear_friction * t).exp();
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
        vx = -vx;
    }
    if y > 1.0 {
        y = 2.0 - y;
        vy = -vy;
    }
    ([x, y], [vx, vy])
}

fn calculate_speed_at_time(speed: f32, acceleration: f32, friction: f32, time: f32) -> f32 {
    let exp = (-1.0 * friction * time).exp();
    exp * (acceleration * (exp - 1.0) + friction * speed)
}

fn acceleration_needed(target: f32, pos: f32, speed: f32, friction: f32, time: f32) -> f32 {
    let d = target - pos;
    let gamma = 1.0 - (-1.0 * friction * time).exp();
    let nom = d - speed * gamma / friction;
    let denom = (friction * time - gamma) / (friction * friction);
    nom / denom
}

fn acceleration_needed_n(target_error: f32, speed: f32, friction: f32, time: f32, n: f32) -> f32 {
    let a_0 = acceleration_needed(target_error, 0.0, speed, friction, time);
    let gamma = 1.0 - (-1.0 * friction * time).exp();
    let nom = friction * n * PI;
    let denom = time - gamma / friction;

    a_0 + nom / denom
}

fn calc_max_allowed_n(
    target_error: f32,
    speed: f32,
    friction: f32,
    time: f32,
    max_acc: f32,
) -> (f32, f32) {
    let a_0 = acceleration_needed(target_error, 0.0, speed, friction, time);
    let gamma = 1.0 - (-1.0 * friction * time).exp();

    let factor = (time - gamma / friction) / (friction * PI);
    let n_low = (-max_acc - a_0) * factor;
    let n_high = (max_acc - a_0) * factor;
    (n_low.ceil(), n_high.floor())
}

fn ball_is_reachable(
    ball_pos: [f32; 2],
    ball_speed: [f32; 2],
    pos: [f32; 2],
    speed: [f32; 2],
    friction: f32,
    max_acc: f32,
    player_height: f32,
    time: f32,
) -> bool {
    let (target_pos, _target_speed) = calculate_state_at_time(ball_pos, ball_speed, 0.0, time, 1.0);
    let ax_low = acceleration_needed(
        target_pos[0] - player_height / 4.0,
        pos[0],
        speed[0],
        friction,
        time,
    );
    let ax_high = acceleration_needed(
        target_pos[0] + player_height / 4.0,
        pos[0],
        speed[0],
        friction,
        time,
    );
    let ay_low = acceleration_needed(
        target_pos[1] - player_height / 4.0,
        pos[1],
        speed[1],
        friction,
        time,
    );
    let ay_high = acceleration_needed(
        target_pos[1] + player_height / 4.0,
        pos[1],
        speed[1],
        friction,
        time,
    );

    return (ax_low.abs().min(ax_high.abs()) < max_acc)
        && (ay_low.abs().min(ay_high.abs()) < max_acc);
}

fn solve_acceleration(
    ball_pos: [f32; 2],
    ball_speed: [f32; 2],
    pos: [f32; 2],
    speed: [f32; 2],
    angle: f32,
    angular_velocity: f32,
    angular_acc: f32,
    time: f32,
    linear_friction: f32,
    angular_friction: f32,
    player_height: f32,
    num_iterations: usize,
) -> (f32, f32, f32) {
    let (target_pos, target_speed) = calculate_state_at_time(ball_pos, ball_speed, 0.0, time, 1.0);
    let mut left_right = 0.0;
    let mut down_up = 0.0;
    let mut angle_incidence = 0.0;
    let mut target_angle = 0.0;
    let mut adjustment_x = 0.0;
    let mut adjustment_y = 0.0;

    for _ in 0..num_iterations {
        (left_right, down_up) = (
            acceleration_needed(
                target_pos[0] + adjustment_x,
                pos[0],
                speed[0],
                linear_friction,
                time,
            ),
            acceleration_needed(
                target_pos[1] + adjustment_y,
                pos[1],
                speed[1],
                linear_friction,
                time,
            ),
        );

        let (speed_x_forecast, speed_y_forecast) = (
            calculate_speed_at_time(speed[0], left_right, linear_friction, time),
            calculate_speed_at_time(speed[1], down_up, linear_friction, time),
        );
        angle_incidence =
            (target_speed[1] - speed_y_forecast).atan2(target_speed[0] - speed_x_forecast);

        target_angle = angle_incidence / 2.0;

        (adjustment_x, adjustment_y) = (
            target_angle.cos() * player_height / 3.0,
            target_angle.sin() * player_height / 3.0,
        );
    }

    let mut angle_error = target_angle - angle;
    while angle_error.abs() > PI / 2.0 {
        if angle_error > 0.0 {
            angle_error -= PI;
        } else {
            angle_error += PI;
        }
    }

    let (n_low, n_high) = calc_max_allowed_n(
        angle_error,
        angular_velocity,
        angular_friction,
        time,
        angular_acc,
    );
    let n = if angle_incidence > 0.0 { n_high } else { n_low };
    let rotate = acceleration_needed_n(angle_error, angular_velocity, angular_friction, time, n);
    (left_right, down_up, rotate)
}

fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    } else if n == 1 {
        return vec![start];
    }

    let step = (end - start) / (n - 1) as f32;
    (0..n - 1).map(|i| start + step * i as f32).collect()
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
            left_right: (rand::random::<f32>() * 2.0 - 1.0).round(),
            down_up: (rand::random::<f32>() * 2.0 - 1.0).round(),
            rotate: (rand::random::<f32>() * 2.0 - 1.0).round(),
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
        let opponent;
        if self.player_id == 1 {
            player = state.get_player1();
            opponent = state.get_player2();
        } else {
            player = state.get_player2();
            opponent = state.get_player1();
        }

        let mut pos = player.pos;
        let mut speed = player.speed;
        let mut angle = player.angle;
        let mut angular_velocity = player.angular_velocity;
        let mut min_x = player.min_x;
        let mut max_x = player.max_x;

        if self.player_id == 1 {
            pos[0] = 1.0 - pos[0];
            ball_pos[0] = 1.0 - ball_pos[0];
            ball_speed[0] = -ball_speed[0];
            speed[0] = -speed[0];
            angle = -angle;
            angular_velocity = -angular_velocity;
            min_x = 1.0 - min_x;
            max_x = 1.0 - max_x;
        }

        // Simulate ball and player movement
        let n_intercepts = 30;
        let x_intercepts = linspace(min_x, max_x, n_intercepts);
        let mut min_time_to_ball = f32::INFINITY;
        for x_intercept in x_intercepts {
            let time_to_ball =
                calculate_time_to_collision_x(ball_pos, ball_speed, 0.0, x_intercept);
            let can_reach = if time_to_ball.is_finite() {
                ball_is_reachable(
                    ball_pos,
                    ball_speed,
                    pos,
                    speed,
                    player.linear_friction,
                    player.linear_acc,
                    player.height,
                    time_to_ball,
                )
            } else {
                false
            };
            if can_reach {
                min_time_to_ball = min_time_to_ball.min(time_to_ball);
            }
        }
        let time_to_ball = min_time_to_ball;

        let left_right;
        let down_up;
        let rotate;

        // If long wait, mirror opponent and stabilize close to center
        if time_to_ball.is_infinite() || time_to_ball > 10.0 {
            let time_to_ball = 1.0;
            left_right = acceleration_needed(
                (min_x + max_x) / 2.0,
                pos[0],
                speed[0],
                player.linear_friction,
                time_to_ball,
            );
            down_up = acceleration_needed(
                (0.5 + opponent.pos[1]) / 2.0,
                pos[1],
                speed[1],
                player.linear_friction,
                time_to_ball,
            );
            let mut target_angle = PI / 2.0 - angle;
            while target_angle.abs() > PI / 2.0 {
                if target_angle > 0.0 {
                    target_angle -= PI;
                } else {
                    target_angle += PI;
                }
            }
            rotate = acceleration_needed(
                target_angle,
                0.0,
                angular_velocity,
                player.angular_friction,
                time_to_ball,
            );
        } else {
            // Try to hit the ball
            let num_iterations = 5;
            (left_right, down_up, rotate) = solve_acceleration(
                ball_pos,
                ball_speed,
                pos,
                speed,
                angle,
                angular_velocity,
                player.angular_acc,
                time_to_ball,
                player.linear_friction,
                player.angular_friction,
                player.height,
                num_iterations,
            );
        }

        let action = if self.player_id == 1 {
            Action {
                left_right: -left_right,
                down_up,
                rotate: -rotate,
            }
        } else {
            Action {
                left_right,
                down_up,
                rotate,
            }
        };
        action
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

pub async fn run_interactive_mode(player: String, opponent: String) {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);

    set_pc_assets_folder("assets");
    let music = load_sound("arcade_music.wav").await.unwrap();

    let soundtrack_params = PlaySoundParams {
        volume: 0.2,
        looped: true,
    };
    play_sound(&music, soundtrack_params);
    let agent1 = create_agent(1, player);
    let agent2 = create_agent(2, opponent);

    let wall_bounce_sound = load_sound("jump1.wav").await.unwrap();
    let paddle_bounce_sound = load_sound("jump2.wav").await.unwrap();

    let renderer = WindowRenderer::new(wall_bounce_sound, paddle_bounce_sound);

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
