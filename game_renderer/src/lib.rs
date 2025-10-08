use game_engine::{Ball, GameState, Player};
use macroquad::prelude::*;

pub trait Renderer {
    fn render(&self, game_state: &GameState);
}

fn draw_player(player: &Player) {
    let pos = player.get_pos();
    let x = pos.x * screen_width();
    let y = pos.y * screen_height();
    let height = player.get_height() * screen_height();
    let width = player.get_width() * screen_height();
    let angle = player.get_angle();
    draw_line(
        x - height * angle.cos() / 2.0,
        y - height * angle.sin() / 2.0,
        x + height * angle.cos() / 2.0,
        y + height * angle.sin() / 2.0,
        width,
        WHITE,
    );
}

fn draw_ball(ball: &Ball) {
    let pos = ball.get_pos();
    let x = pos.x * screen_width();
    let y = pos.y * screen_height();
    let radius = ball.get_radius() * screen_width();
    draw_circle(x, y, radius, WHITE);
}

pub struct WindowRenderer {}

impl WindowRenderer {
    pub fn new() -> Self {
        WindowRenderer {}
    }
}

impl Renderer for WindowRenderer {
    fn render(&self, game_state: &GameState) {
        clear_background(BLACK);

        draw_player(game_state.get_player1());
        draw_player(game_state.get_player2());
        draw_ball(game_state.get_ball());
    }
}
