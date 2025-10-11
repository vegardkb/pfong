use clap::Parser;
use game_modes::{run_agent_headless_mode, run_agent_headless_training_mode, run_interactive_mode};
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

    /// Opponent name
    #[arg(short, long, default_value_t = ("Python").to_string())]
    opponent: String,

    /// Step time between frames
    #[arg(short, long, default_value_t = 0.05)]
    time_step: f32,

    /// Number of games to play
    #[arg(long, default_value_t = 1)]
    num_games: u32,

    /// Number of cycles to train for
    #[arg(long, default_value_t = 1000)]
    num_cycles: u32,

    /// Number of training games to play
    #[arg(long, default_value_t = 2)]
    num_training: u32,

    /// Number of validation games to play
    #[arg(long, default_value_t = 2)]
    num_validation: u32,

    /// Player 1 name
    #[arg(long, default_value_t = ("Python").to_string())]
    player_1: String,

    /// Player 2 name
    #[arg(long, default_value_t = ("Python").to_string())]
    player_2: String,
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();

    let opponents = vec!["Random".to_string()];
    match args.mode.as_str() {
        "interactive" => run_interactive_mode(args.opponent).await,
        "headless" => run_agent_headless_mode(args.time_step, args.num_games).await,
        "training" => {
            run_agent_headless_training_mode(
                args.time_step,
                args.num_cycles,
                args.num_training,
                args.num_validation,
                &opponents,
                &args.player_1,
                &args.player_2,
            )
            .await
        }
        _ => panic!("Invalid mode"),
    }
}
