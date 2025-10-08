use macroquad::math::{Vec2, vec2};
use std::f32::consts::PI;
use std::time::Instant;

pub struct GameEngine {
    state: GameState,
    config: GameConfig,
}

pub struct GameState {
    player1: Player,
    player2: Player,
    ball: Ball,
    score: [u32; 2],
    hit_time: Instant,
    point_time: Instant,
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
    ball_speed: Vec2,
    ball_mass: f32,
}

pub struct Action {
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub rotate_left: bool,
    pub rotate_right: bool,
}

pub struct Ball {
    pos: Vec2,
    speed: Vec2,
    radius: f32,
    mass: f32,
}

pub struct Player {
    pos: Vec2,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    width: f32,
    height: f32,
    speed: Vec2,
    linear_acc: f32,
    linear_friction: f32,
    angle: f32,
    angular_velocity: f32,
    angular_acc: f32,
    angular_friction: f32,
    mass: f32,
}

impl GameEngine {
    pub fn new(config: GameConfig) -> Self {
        GameEngine {
            state: GameState::new(&config),
            config: config,
        }
    }

    pub fn reset(&mut self) {
        self.state = GameState::new(&self.config);
    }

    pub fn step(&mut self, actions: [Action; 2], delta_time: f32) {
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
            hit_time: Instant::now(),
            point_time: Instant::now(),
        }
    }

    fn step(&mut self, actions: [Action; 2], delta_time: f32) {
        self.player1.input(&actions[0], delta_time);
        self.player2.input(&actions[1], delta_time);
        if self.hit_time.elapsed().as_millis() > 50 {
            if self.ball.check_collision(&mut self.player1)
                | self.ball.check_collision(&mut self.player2)
            {
                self.hit_time = Instant::now();
            }
        }

        self.player1.update(delta_time);
        self.player2.update(delta_time);

        let wall_hit = self.ball.update(delta_time);
        if self.point_time.elapsed().as_millis() > 100 {
            if wall_hit == 1 {
                self.score[0] += 1;
                self.point_time = Instant::now();
            } else if wall_hit == 2 {
                self.score[1] += 1;
                self.point_time = Instant::now();
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
}

impl GameConfig {
    pub fn default() -> Self {
        GameConfig {
            player_width: 0.01,
            player_height: 0.15,
            player_linear_acc: 0.7,
            player_linear_friction: 0.4,
            player_angular_acc: 10.0,
            player_angular_friction: 0.4,
            player_mass: 2.0,
            ball_radius: 0.01,
            ball_speed: vec2(0.1, 0.01),
            ball_mass: 1.0,
        }
    }
}

impl Ball {
    fn new(config: &GameConfig) -> Self {
        Ball {
            pos: vec2(0.5, 0.5),
            speed: config.ball_speed,
            radius: config.ball_radius,
            mass: config.ball_mass,
        }
    }

    fn update(&mut self, delta_time: f32) -> u8 {
        if self.pos.x - self.radius < 0.0 {
            self.speed.x = self.speed.x.abs();
        }
        if self.pos.x + self.radius > 1.0 {
            self.speed.x = -self.speed.x.abs();
        }
        if self.pos.y - self.radius < 0.0 {
            self.speed.y = self.speed.y.abs();
        }
        if self.pos.y + self.radius > 1.0 {
            self.speed.y = -self.speed.y.abs();
        }

        let wall_hit = if self.pos.x - self.radius < 0.0 {
            1
        } else if self.pos.x + self.radius > 1.0 {
            2
        } else {
            0
        };

        self.pos.x += self.speed.x * delta_time;
        self.pos.y += self.speed.y * delta_time;

        wall_hit
    }

    fn check_collision(&mut self, player: &mut Player) -> bool {
        let dc = self.pos - player.pos;
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
            let v_c = player.speed + player.angular_velocity * d * n;

            let v_rel = self.speed - v_c;

            let i = player.mass * player.height * player.height / 12.0;
            let j = -2.0 * v_rel.dot(n) / (1.0 / self.mass + 1.0 / player.mass + (d * d) / i);

            self.speed += j * n / self.mass;

            player.speed -= j * n / player.mass;
            player.angular_velocity -= j * d / i;
            out = true;
        }
        out
    }

    pub fn get_pos(&self) -> Vec2 {
        self.pos
    }

    pub fn get_radius(&self) -> f32 {
        self.radius
    }
}

impl Player {
    fn new(config: &GameConfig, player_id: u32) -> Self {
        let xpos = if player_id == 1 { 0.8 } else { 0.2 };
        let min_x = if player_id == 1 { 0.6 } else { 0.0 };
        let max_x = if player_id == 1 { 1.0 } else { 0.4 };
        let min_y = 0.0;
        let max_y = 1.0;

        Player {
            pos: vec2(xpos, 0.5),
            speed: vec2(0.0, 0.0),
            angular_velocity: 0.0,
            mass: config.player_mass,
            width: config.player_width,
            height: config.player_height,
            angle: PI / 2.0,
            linear_acc: config.player_linear_acc,
            linear_friction: config.player_linear_friction,
            angular_acc: config.player_angular_acc,
            angular_friction: config.player_angular_friction,
            min_x: min_x,
            max_x: max_x,
            min_y: min_y,
            max_y: max_y,
        }
    }

    fn input(&mut self, action: &Action, delta_time: f32) {
        if action.up {
            self.speed.y -= self.linear_acc * delta_time;
        }
        if action.down {
            self.speed.y += self.linear_acc * delta_time;
        }
        if action.left {
            self.speed.x -= self.linear_acc * delta_time;
        }
        if action.right {
            self.speed.x += self.linear_acc * delta_time;
        }
        if action.rotate_left {
            self.angular_velocity -= self.angular_acc * delta_time;
        }
        if action.rotate_right {
            self.angular_velocity += self.angular_acc * delta_time;
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.pos.x += self.speed.x * delta_time;
        self.pos.y += self.speed.y * delta_time;
        self.angle += self.angular_velocity * delta_time;
        if self.angle > 2.0 * PI {
            self.angle -= 2.0 * PI;
        }
        if self.angle < 0.0 {
            self.angle += 2.0 * PI;
        }

        if self.pos.x < self.min_x {
            self.pos.x = self.min_x;
            self.speed.x = -self.speed.x;
        } else if self.pos.x > self.max_x {
            self.pos.x = self.max_x;
            self.speed.x = -self.speed.x;
        } else if self.pos.y < self.min_y {
            self.pos.y = self.min_y;
            self.speed.y = -self.speed.y;
        } else if self.pos.y > self.max_y {
            self.pos.y = self.max_y;
            self.speed.y = -self.speed.y;
        }

        self.angular_velocity -= self.angular_friction * self.angular_velocity * delta_time;
        self.speed -= self.linear_friction * self.speed * delta_time;
    }

    pub fn get_pos(&self) -> Vec2 {
        self.pos
    }

    pub fn get_height(&self) -> f32 {
        self.height
    }

    pub fn get_width(&self) -> f32 {
        self.width
    }

    pub fn get_angle(&self) -> f32 {
        self.angle
    }
}
