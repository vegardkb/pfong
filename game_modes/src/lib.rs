use game_engine::{Action, GameConfig, GameEngine, GameState};
use game_renderer::{Renderer, WindowRenderer};
use macroquad::input::{KeyCode, is_key_down};
use macroquad::window::next_frame;
use std::time::Instant;

pub struct GameSession<T: Renderer> {
    engine: GameEngine,
    renderer: T,
    agent1: Box<dyn Agent>,
    agent2: Box<dyn Agent>,
}

impl GameSession<WindowRenderer> {
    pub async fn run(&mut self) {
        let mut step_time = Instant::now();

        loop {
            let state = self.engine.get_state();

            let actions = [self.agent1.get_action(state), self.agent2.get_action(state)];

            self.engine.step(actions, step_time.elapsed().as_secs_f32());
            step_time = Instant::now();

            let state = self.engine.get_state();
            self.renderer.render(&state);

            if state.is_terminal() {
                break;
            }

            next_frame().await;
        }
    }
}

pub trait Agent {
    fn get_action(&mut self, state: &GameState) -> Action;
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
            pos.x = 1.0 - pos.x;
            ball_pos.x = 1.0 - ball_pos.x;
        } else {
            pos = state.get_player2().get_pos();
        }

        if ball_pos.x > pos.x {
            if ball_pos.y > pos.y {
                down = true;
            } else if ball_pos.y < pos.y {
                up = true;
            }
        } else if ball_pos.y > pos.y {
            up = true;
        } else if ball_pos.y < pos.y {
            down = true;
        }

        Action {
            left: false,
            right: false,
            up: up,
            down: down,
            rotate_left: false,
            rotate_right: false,
        }
    }
}

pub async fn run_interactive_mode() {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);
    let renderer = WindowRenderer::new();
    let agent1 = Box::new(HumanAgent {});
    let agent2 = Box::new(HeuristicAgent { player_id: 2 });
    let mut session = GameSession {
        engine,
        renderer,
        agent1,
        agent2,
    };
    session.run().await;
}
