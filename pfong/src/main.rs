use game_modes::run_interactive_mode;
use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "pfong".to_owned(),
        fullscreen: false,
        window_height: 800,
        window_width: 800,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    run_interactive_mode().await;
}
