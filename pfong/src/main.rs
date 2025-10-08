use clap::Parser;
use game_modes::{run_agent_headless_mode, run_interactive_mode};
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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Game mode
    #[arg(short, long, default_value_t = ("interactive").to_string())]
    mode: String,

    /// Step time between frames
    #[arg(short, long, default_value_t = 0.1)]
    time_step: f32,
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();

    match args.mode.as_str() {
        "interactive" => run_interactive_mode().await,
        "headless" => run_agent_headless_mode(args.time_step).await,
        _ => panic!("Invalid mode"),
    }
}
