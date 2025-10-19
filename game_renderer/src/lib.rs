use game_engine::{Ball, GameState, Player};
use macroquad::prelude::*;

pub trait Renderer {
    fn render(&self, game_state: &GameState);
}

fn draw_player(player: &Player) {
    let pos = player.pos;
    let x = pos[0] * screen_width();
    let y = pos[1] * screen_height();
    let height = player.height * screen_height();
    let width = player.width * screen_height();
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

fn draw_ball(ball: &Ball) {
    let pos = ball.pos;
    let x = pos[0] * screen_width();
    let y = pos[1] * screen_height();
    let radius = ball.radius * screen_width();
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
