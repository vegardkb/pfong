use game_engine::{GameConfig, GameEngine};
use game_renderer::{HeadlessRenderer, Renderer, WindowRenderer};
use macroquad::audio::{PlaySoundParams, load_sound, play_sound};
use macroquad::prelude::set_pc_assets_folder;
use macroquad::window::next_frame;
use pfong_agent::{Agent, SessionMetadata, create_agent};
use std::time::Instant;

mod menu;
use menu::{ButtonAction, GameStateType, MenuSystem};

pub struct GameSession<T: Renderer> {
    engine: GameEngine,
    renderer: T,
    agent1: Box<dyn Agent>,
    agent2: Box<dyn Agent>,
    step_count: u32,
    menu: MenuSystem,
}

impl GameSession<WindowRenderer> {
    pub async fn run(&mut self, evaluation_mode: bool) {
        let mut step_time = Instant::now();

        let metadata = SessionMetadata {
            evaluation_mode,
            player1_id: self.agent1.get_name(),
            player2_id: self.agent2.get_name(),
        };

        loop {
            if let Some(action) = self.menu.update() {
                match action {
                    ButtonAction::StartGame | ButtonAction::NewGame => {
                        self.menu.set_state(GameStateType::Playing);
                        self.engine.reset();
                        self.step_count = 0;
                        step_time = Instant::now();
                    }
                    ButtonAction::Quit => {
                        break;
                    }
                }
            }

            match self.menu.get_state() {
                GameStateType::Playing => {
                    let state = self.engine.get_state();
                    let actions = (
                        self.agent1.get_action(state, &metadata),
                        self.agent2.get_action(state, &metadata),
                    );

                    self.engine.step(actions, step_time.elapsed().as_secs_f32());
                    step_time = Instant::now();
                    self.step_count += 1;

                    let state = self.engine.get_state();
                    if state.is_terminal() {
                        let _ = (
                            self.agent1.get_action(state, &metadata),
                            self.agent2.get_action(state, &metadata),
                        );

                        let score = state.get_score();
                        let winner = if score[0] >= 10 {
                            "Player 1".to_string()
                        } else {
                            "Player 2".to_string()
                        };

                        self.menu.set_state(GameStateType::GameOver { winner });
                    }
                }
                GameStateType::GameOver { .. } => {
                    let actions = (
                        game_engine::Action::default(),
                        game_engine::Action::default(),
                    );
                    self.engine.step(actions, step_time.elapsed().as_secs_f32());
                    step_time = Instant::now();
                }
                GameStateType::Welcome => {
                }
            }

            let state = self.engine.get_state();
            self.renderer.render(state);

            // Render menu overlay
            self.menu.render();

            next_frame().await;
        }
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

pub async fn run_interactive_mode(player: String, opponent: String) {
    let config = GameConfig::default();
    let engine = GameEngine::new(config);

    set_pc_assets_folder("assets");
    let music = match load_sound("arcade_music.wav").await {
        Ok(sound) => sound,
        Err(err) => {
            eprintln!("Failed to load sound: {err}");
            return;
        }
    };

    let soundtrack_params = PlaySoundParams {
        volume: 0.5,
        looped: true,
    };
    play_sound(&music, soundtrack_params);
    let agent1 = create_agent(1, player);
    let agent2 = create_agent(2, opponent);

    let wall_bounce_sound = match load_sound("pingpong_paddle_trim.wav").await {
        Ok(sound) => sound,
        Err(err) => {
            eprintln!("Failed to load sound: {err}");
            return;
        }
    };
    let paddle_bounce_sound = match load_sound("pingpong_paddle_trim.wav").await {
        Ok(sound) => sound,
        Err(err) => {
            eprintln!("Failed to load sound: {err}");
            return;
        }
    };
    let player_scored_sound = match load_sound("point_for.wav").await {
        Ok(sound) => sound,
        Err(err) => {
            eprintln!("Failed to load sound: {err}");
            return;
        }
    };
    let opponent_scored_sound = match load_sound("point_against.wav").await {
        Ok(sound) => sound,
        Err(err) => {
            eprintln!("Failed to load sound: {err}");
            return;
        }
    };

    let renderer = WindowRenderer::new(
        wall_bounce_sound,
        paddle_bounce_sound,
        player_scored_sound,
        opponent_scored_sound,
    );

    let step_count = 0;
    let mut session = GameSession {
        engine,
        renderer,
        agent1,
        agent2,
        step_count,
        menu: MenuSystem::new(),
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
            menu: MenuSystem::new(),
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
                menu: MenuSystem::new(),
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
                    menu: MenuSystem::new(),
                };
                let evaluation_mode = true;
                session.run(delta_time, evaluation_mode).await;
            }
        }
    }
}
