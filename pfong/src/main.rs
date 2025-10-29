use clap::Parser;
use game_modes::run_interactive_mode;
use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "pfong".to_owned(),
        fullscreen: true,
        ..Default::default()
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Player name
    #[arg(short, long, default_value_t = ("Keyboard").to_string())]
    player: String,

    /// Opponent name
    #[arg(short, long, default_value_t = ("Heuristic").to_string())]
    opponent: String,
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();

    run_interactive_mode(args.player, args.opponent).await
}
