use std::fs;

use rfd::AsyncFileDialog;

pub mod dtos;
pub mod entities;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
async fn save_genome(genome_str: &str) -> Result<(), String> {
    let file = AsyncFileDialog::new()
        .set_title("Save genome as...")
        .add_filter("Genome", &["evog"])
        .save_file()
        .await;


    if let Some(file_handle) = file {
        fs::write(file_handle.path(), genome_str).or_else(|e| Err(e.to_string()))
    }
    else {
        Ok(())
    }
}

#[tauri::command]
async fn load_genome() -> Result<String, String> {
    let file = AsyncFileDialog::new()
        .set_title("Select genome file to upload")
        .add_filter("Genome", &["evog"])
        .pick_file()
        .await;

    if let Some(file_handle) = file {
        fs::read_to_string(file_handle.path()).or_else(|e| Err(e.to_string()))
    }
    else {
        Ok("".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![save_genome, load_genome])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
