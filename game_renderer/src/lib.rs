use game_engine::{Ball, GameState, Player};
use macroquad::audio::{PlaySoundParams, play_sound};
use macroquad::prelude::*;

pub trait Renderer {
    fn render(&self, game_state: &GameState);
}

pub struct WindowRenderer {
    wall_bounce_sound: macroquad::audio::Sound,
    paddle_bounce_sound: macroquad::audio::Sound,
    player_scored_sound: macroquad::audio::Sound,
    opponent_scored_sound: macroquad::audio::Sound,
}

impl WindowRenderer {
    pub fn new(
        wall_bounce_sound: macroquad::audio::Sound,
        paddle_bounce_sound: macroquad::audio::Sound,
        player_scored_sound: macroquad::audio::Sound,
        opponent_scored_sound: macroquad::audio::Sound,
    ) -> Self {
        WindowRenderer {
            wall_bounce_sound,
            paddle_bounce_sound,
            player_scored_sound,
            opponent_scored_sound,
        }
    }

    fn pos_to_coordinates(&self, pos: [f32; 2]) -> (f32, f32) {
        let shortest_side = screen_width().min(screen_height());
        let pad_x = (screen_width() - shortest_side) / 2.0;
        let pad_y = (screen_height() - shortest_side) / 2.0;
        let x = (pos[0] * 0.8 + 0.1) * shortest_side + pad_x;
        let y = (pos[1] * 0.8 + 0.15) * shortest_side + pad_y;
        (x, y)
    }

    fn scale_dimension(&self, length: f32) -> f32 {
        length * screen_height() * 0.8
    }

    fn draw_player(&self, player: &Player) {
        let pos = player.pos;
        let (x, y) = self.pos_to_coordinates(pos);
        let height = self.scale_dimension(player.height);
        let width = self.scale_dimension(player.width);
        let angle = player.angle;
        draw_line(
            x - height * angle.cos() / 2.0,
            y - height * angle.sin() / 2.0,
            x + height * angle.cos() / 2.0,
            y + height * angle.sin() / 2.0,
            width,
            WHITE,
        );
    }

    fn draw_ball(&self, ball: &Ball) {
        let pos = ball.pos;
        let (x, y) = self.pos_to_coordinates(pos);
        let radius = self.scale_dimension(ball.radius);

        let color = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        draw_circle(x, y, radius, color);
    }

    fn draw_outer_boundary(&self) {
        let width = self.scale_dimension(1.0);
        let height = self.scale_dimension(1.0);
        let (x, y) = self.pos_to_coordinates([0.0, 0.0]);
        draw_rectangle_lines(x, y, width, height, 3.0, WHITE);
    }

    fn draw_inner_boundary(&self, player: &Player) {
        let (x, y) = self.pos_to_coordinates([player.min_x, player.min_y]);
        let (x1, y1) = self.pos_to_coordinates([player.max_x, player.max_y]);
        let w = x1 - x;
        let h = y1 - y;
        draw_rectangle_lines(x, y, w, h, 1.5, WHITE);
    }

    fn draw_score(&self, game_state: &GameState) {
        let score = game_state.get_score();
        let font_size = 20.0;
        let (x, y) = self.pos_to_coordinates([0.88, -0.05]);
        draw_text(&format!("Player 1: {}", score[0]), x, y, font_size, WHITE);
        let (x, y) = self.pos_to_coordinates([0.0, -0.05]);
        draw_text(&format!("Player 2: {}", score[1]), x, y, font_size, WHITE);
    }
}

impl Renderer for WindowRenderer {
    fn render(&self, game_state: &GameState) {
        let last_point_scored = game_state.get_last_point_scored();

        let player1 = game_state.get_player1();
        let player2 = game_state.get_player2();
        let ball = game_state.get_ball();
        let max_energy = 7.0;

        let mut r = player2.calc_energy().clamp(0.0, max_energy) / max_energy;
        let g = ball.calc_energy().clamp(0.0, max_energy) / max_energy;
        let mut b = player1.calc_energy().clamp(0.0, max_energy) / max_energy;

        let wall_factor = 4.0;
        let paddle_factor = 4.0;

        if game_state.get_wall_hit() {
            let volume = wall_factor * game_state.get_impulse().abs().clamp(0.0, 1.0);
            let sound_params = PlaySoundParams {
                volume,
                looped: false,
            };
            play_sound(&self.wall_bounce_sound, sound_params);
        }

        if game_state.get_ball_hit() != 0 {
            let impulse = game_state.get_impulse().abs().clamp(0.0, 1.0);
            let volume = paddle_factor * impulse;
            let sound_params = PlaySoundParams {
                volume,
                looped: false,
            };
            play_sound(&self.paddle_bounce_sound, sound_params);
        }

        let point_scored = game_state.get_point_scored();
        if point_scored == 1 {
            play_sound(
                &self.player_scored_sound,
                PlaySoundParams {
                    looped: false,
                    volume: 1.0,
                },
            )
        } else if point_scored == 2 {
            play_sound(
                &self.opponent_scored_sound,
                PlaySoundParams {
                    looped: false,
                    volume: 1.0,
                },
            )
        }

        let point_time_scale = 0.2;
        let point_intensity = 0.5;
        if last_point_scored == 1 {
            let point_time = game_state.get_point_time();
            b += point_intensity * point_time_scale / (point_time + point_time_scale);
        } else if last_point_scored == 2 {
            let point_time = game_state.get_point_time();
            r += point_intensity * point_time_scale / (point_time + point_time_scale);
        }

        let score_intensity = 0.05;
        let score = game_state.get_score();
        b += score[0] as f32 * score_intensity;
        r += score[1] as f32 * score_intensity;

        clear_background(Color { r, g, b, a: 0.5 });

        self.draw_player(player1);
        self.draw_player(player2);
        self.draw_ball(ball);

        self.draw_outer_boundary();
        self.draw_inner_boundary(player1);
        self.draw_inner_boundary(player2);
        self.draw_score(game_state);
    }
}

pub struct HeadlessRenderer {}

impl HeadlessRenderer {
    pub fn new() -> Self {
        HeadlessRenderer {}
    }
}

impl Renderer for HeadlessRenderer {
    fn render(&self, _game_state: &GameState) {
        // No rendering in headless mode
    }
}

impl Default for HeadlessRenderer {
    fn default() -> Self {
        HeadlessRenderer::new()
    }
}

pub fn get_acc_pad_pos() -> (f32, f32, f32, f32) {
    let height = screen_height();
    let width = screen_width();
    let w = height.min(width) / 5.0;
    let x = width / 20.0;
    let y = 19.0 * height / 20.0 - w;
    (x, y, w, w)
}

pub fn get_rot_pad_pos() -> (f32, f32, f32, f32) {
    let height = screen_height();
    let width = screen_width();
    let min_dim = height.min(width);
    let w = min_dim / 5.0;
    let h = min_dim / 20.0;
    let x = 19.0 * width / 20.0 - w;
    let y = 19.0 * height / 20.0 - 2.0 * h;
    (x, y, w, h)
}
