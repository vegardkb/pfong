use game_engine::{Ball, GameState, Player};
use macroquad::prelude::*;

pub trait Renderer {
    fn render(&self, game_state: &GameState);
}

fn get_state_energy(game_state: &GameState) -> f32 {
    game_state.get_player1().calc_energy()
        + game_state.get_player2().calc_energy()
        + game_state.get_ball().calc_energy()
}

pub struct WindowRenderer {}

impl WindowRenderer {
    pub fn new() -> Self {
        WindowRenderer {}
    }

    fn pos_to_coordinates(&self, pos: [f32; 2]) -> (f32, f32) {
        let x = (pos[0] * 0.8 + 0.1) * screen_width();
        let y = (pos[1] * 0.8 + 0.15) * screen_height();
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
        let energy = get_state_energy(game_state);
        println!("Energy: {}", energy);
        let max_energy = 2.5;

        let mut r = player2.calc_energy().clamp(0.0, max_energy) / max_energy;
        let g = ball.calc_energy().clamp(0.0, max_energy) / max_energy;
        let mut b = player1.calc_energy().clamp(0.0, max_energy) / max_energy;

        let point_time_scale = 0.2;
        let point_intensity = 0.5;
        if last_point_scored == 1 {
            let point_time = game_state.get_point_time();
            b = b + point_intensity * point_time_scale / (point_time + point_time_scale);
        } else if last_point_scored == 2 {
            let point_time = game_state.get_point_time();
            r = r + point_intensity * point_time_scale / (point_time + point_time_scale);
        }

        clear_background(Color { r, g, b, a: 0.5 });
        //clear_background(BLACK);

        self.draw_player(player1);
        self.draw_player(player2);
        self.draw_ball(ball);

        self.draw_outer_boundary();
        self.draw_inner_boundary(player1);
        self.draw_inner_boundary(player2);
        self.draw_score(game_state);
    }
}

impl Default for WindowRenderer {
    fn default() -> Self {
        WindowRenderer::new()
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
