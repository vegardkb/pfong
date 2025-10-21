use macroquad::math::vec2;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::fmt::Debug;
use uuid::Uuid;

pub struct GameEngine {
    state: GameState,
    config: GameConfig,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    player1: Player,
    player2: Player,
    ball: Ball,
    score: [u32; 2],
    ball_hit: u8,
    point_scored: u8,
    last_point_scored: u8,
    hit_time: f32,
    point_time: f32,
    elapsed_time: f32,
    game_uuid: String,
}

pub struct GameConfig {
    player_width: f32,
    player_height: f32,
    player_linear_acc: f32,
    player_linear_friction: f32,
    player_angular_acc: f32,
    player_angular_friction: f32,
    player_mass: f32,
    ball_radius: f32,
    ball_speed: [f32; 2],
    ball_mass: f32,
}

#[derive(Clone, Serialize, Deserialize, Default, Copy, Debug)]
pub struct Action {
    pub left_right: f32,
    pub down_up: f32,
    pub rotate: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Ball {
    pub pos: [f32; 2],
    pub speed: [f32; 2],
    pub radius: f32,
    pub mass: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Player {
    pub pos: [f32; 2],
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub width: f32,
    pub height: f32,
    pub speed: [f32; 2],
    pub linear_acc: f32,
    pub linear_friction: f32,
    pub angle: f32,
    pub angular_velocity: f32,
    pub angular_acc: f32,
    pub angular_friction: f32,
    pub mass: f32,
}

impl GameEngine {
    pub fn new(config: GameConfig) -> Self {
        GameEngine {
            state: GameState::new(&config),
            config,
        }
    }

    pub fn reset(&mut self) {
        self.state = GameState::new(&self.config);
    }

    pub fn step(&mut self, actions: (Action, Action), delta_time: f32) {
        self.state.step(actions, delta_time);
    }

    pub fn get_state(&self) -> &GameState {
        &self.state
    }
}

impl GameState {
    fn new(config: &GameConfig) -> Self {
        GameState {
            player1: Player::new(config, 1),
            player2: Player::new(config, 2),
            ball: Ball::new(config),
            score: [0, 0],
            ball_hit: 0,
            point_scored: 0,
            last_point_scored: 0,
            hit_time: 0.0,
            point_time: 0.0,
            elapsed_time: 0.0,
            game_uuid: Uuid::new_v4().to_string(),
        }
    }

    fn step(&mut self, actions: (Action, Action), delta_time: f32) {
        self.step_timers(delta_time);

        self.ball_hit = 0;
        self.point_scored = 0;

        self.check_ball_collision();

        self.player1.update(delta_time);
        self.player2.update(delta_time);
        self.player1.input(&actions.0, delta_time);
        self.player2.input(&actions.1, delta_time);

        self.check_ball_collision();

        let wall_hit = self.ball.update(delta_time);
        self.handle_wall_hit(wall_hit);
    }

    fn step_timers(&mut self, delta_time: f32) {
        self.hit_time += delta_time;
        self.point_time += delta_time;
        self.elapsed_time += delta_time;
    }

    fn check_ball_collision(&mut self) {
        if self.hit_time > 0.05 {
            if self.ball.check_collision(&mut self.player1) {
                self.ball_hit = 1;
                self.hit_time = 0.0;
            } else if self.ball.check_collision(&mut self.player2) {
                self.ball_hit = 2;
                self.hit_time = 0.0;
            }
        }
    }

    fn handle_wall_hit(&mut self, wall_hit: u8) {
        if self.point_time > 0.05 {
            if wall_hit == 1 {
                self.score[0] += 1;
                self.point_time = 0.0;
                self.point_scored = 1;
                self.last_point_scored = 1;
            } else if wall_hit == 2 {
                self.score[1] += 1;
                self.point_time = 0.0;
                self.point_scored = 2;
                self.last_point_scored = 2;
            }
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.score[0] >= 10 || self.score[1] >= 10
    }

    pub fn get_player1(&self) -> &Player {
        &self.player1
    }

    pub fn get_player2(&self) -> &Player {
        &self.player2
    }

    pub fn get_ball(&self) -> &Ball {
        &self.ball
    }

    pub fn get_score(&self) -> &[u32; 2] {
        &self.score
    }

    pub fn get_last_point_scored(&self) -> u8 {
        self.last_point_scored
    }

    pub fn get_point_time(&self) -> f32 {
        self.point_time
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        let mut speed_x = rand::random::<f32>() / 2.0 + 0.02;
        let mut speed_y = rand::random::<f32>() / 2.0 + 0.02;
        speed_x = if rand::random::<bool>() {
            speed_x
        } else {
            -speed_x
        };
        speed_y = if rand::random::<bool>() {
            speed_y
        } else {
            -speed_y
        };
        let speed = [speed_x, speed_y];

        GameConfig {
            player_width: 0.01,
            player_height: 0.10,
            player_linear_acc: 1.0, //1.0,
            player_linear_friction: 0.3,
            player_angular_acc: 10.0, //10.0,
            player_angular_friction: 0.3,
            player_mass: 3.0,
            ball_radius: 0.01,
            ball_speed: speed,
            ball_mass: 1.0,
        }
    }
}

impl Ball {
    fn new(config: &GameConfig) -> Self {
        Ball {
            pos: [0.5, 0.5],
            speed: config.ball_speed,
            radius: config.ball_radius,
            mass: config.ball_mass,
        }
    }

    fn update(&mut self, delta_time: f32) -> u8 {
        if self.pos[0] - self.radius < 0.0 {
            self.speed[0] = self.speed[0].abs();
        }
        if self.pos[0] + self.radius > 1.0 {
            self.speed[0] = -self.speed[0].abs();
        }
        if self.pos[1] - self.radius < 0.0 {
            self.speed[1] = self.speed[1].abs();
        }
        if self.pos[1] + self.radius > 1.0 {
            self.speed[1] = -self.speed[1].abs();
        }

        let wall_hit = if self.pos[0] - self.radius < 0.0 {
            1
        } else if self.pos[0] + self.radius > 1.0 {
            2
        } else {
            0
        };

        self.pos[0] += self.speed[0] * delta_time;
        self.pos[1] += self.speed[1] * delta_time;

        wall_hit
    }

    fn check_collision(&mut self, player: &mut Player) -> bool {
        let ball_pos = vec2(self.pos[0], self.pos[1]);
        let player_pos = vec2(player.pos[0], player.pos[1]);
        let mut ball_speed = vec2(self.speed[0], self.speed[1]);
        let mut player_speed = vec2(player.speed[0], player.speed[1]);

        let dc = ball_pos - player_pos;
        let r = player.height * vec2(player.angle.cos(), player.angle.sin()) / 2.0;
        let d1s = dc + r;
        let d1 = d1s.dot(d1s).sqrt();

        let d2s = dc - r;
        let d2 = d2s.dot(d2s).sqrt();

        let buffer = player.width / 2.0 + self.radius / 2.0;
        let mut out = false;
        if (d1 + d2 > player.height - buffer) & (d1 + d2 < player.height + buffer) {
            let n = vec2(-player.angle.sin(), player.angle.cos());

            let d = d1 * player.height / (d1 + d2) - player.height / 2.0;
            let v_c = player_speed + player.angular_velocity * d * n;

            let v_rel = ball_speed - v_c;

            let i = player.calc_moment_of_inertia();
            let j = -2.0 * v_rel.dot(n) / (1.0 / self.mass + 1.0 / player.mass + (d * d) / i);

            ball_speed += j * n / self.mass;
            self.speed = [ball_speed.x, ball_speed.y];

            player_speed -= j * n / player.mass;
            player.speed = [player_speed.x, player_speed.y];
            player.angular_velocity -= j * d / i;
            out = true;
        }
        out
    }

    pub fn calc_energy(&self) -> f32 {
        self.mass * self.speed[0] * self.speed[0] / 2.0
            + self.mass * self.speed[1] * self.speed[1] / 2.0
    }
}

impl Player {
    fn new(config: &GameConfig, player_id: u32) -> Self {
        let xpos = if player_id == 1 { 0.8 } else { 0.2 };
        let min_x = if player_id == 1 { 0.67 } else { 0.0 };
        let max_x = if player_id == 1 { 1.0 } else { 0.33 };
        let min_y = 0.0;
        let max_y = 1.0;

        Player {
            pos: [xpos, 0.5],
            speed: [0.0, 0.0],
            angular_velocity: 0.0,
            mass: config.player_mass,
            width: config.player_width,
            height: config.player_height,
            angle: PI / 2.0,
            linear_acc: config.player_linear_acc,
            linear_friction: config.player_linear_friction,
            angular_acc: config.player_angular_acc,
            angular_friction: config.player_angular_friction,
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    fn input(&mut self, action: &Action, delta_time: f32) {
        let down_up = if action.down_up.is_nan() {
            0.0
        } else {
            action.down_up.clamp(-1.0, 1.0)
        };
        self.speed[1] += down_up * self.linear_acc * delta_time;
        let left_right = if action.left_right.is_nan() {
            0.0
        } else {
            action.left_right.clamp(-1.0, 1.0)
        };
        self.speed[0] += left_right * self.linear_acc * delta_time;
        let rotate = if action.rotate.is_nan() {
            0.0
        } else {
            action.rotate.clamp(-1.0, 1.0)
        };
        self.angular_velocity += rotate * self.angular_acc * delta_time;
    }

    fn update(&mut self, delta_time: f32) {
        self.pos[0] += self.speed[0] * delta_time;
        self.pos[1] += self.speed[1] * delta_time;
        self.angle += self.angular_velocity * delta_time;
        while self.angle.abs() > PI / 2.0 {
            if self.angle > PI / 2.0 {
                self.angle -= PI;
            } else {
                self.angle += PI;
            }
        }

        if self.pos[0] < self.min_x {
            self.pos[0] = self.min_x;
            self.speed[0] = -self.speed[0];
        } else if self.pos[0] > self.max_x {
            self.pos[0] = self.max_x;
            self.speed[0] = -self.speed[0];
        } else if self.pos[1] < self.min_y {
            self.pos[1] = self.min_y;
            self.speed[1] = -self.speed[1];
        } else if self.pos[1] > self.max_y {
            self.pos[1] = self.max_y;
            self.speed[1] = -self.speed[1];
        }

        self.angular_velocity -= self.angular_velocity * self.angular_friction * delta_time;
        self.speed[0] -= self.speed[0] * self.linear_friction * delta_time;
        self.speed[1] -= self.speed[1] * self.linear_friction * delta_time;
    }

    fn calc_moment_of_inertia(&self) -> f32 {
        self.mass * self.height * self.height / 12.0
    }

    pub fn calc_energy(&self) -> f32 {
        self.mass * self.speed[0] * self.speed[0] / 2.0
            + self.mass * self.speed[1] * self.speed[1] / 2.0
            + self.calc_moment_of_inertia() * self.angular_velocity * self.angular_velocity / 2.0
    }
}
