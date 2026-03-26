// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--train-worker") {
        neural_evo_tauri_app_lib::run_train_worker();
    } else {
        neural_evo_tauri_app_lib::run()
    }
}
