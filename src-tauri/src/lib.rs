use std::fs;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn save_genome(genome_str: &str, path_str: &str) -> Result<(), String> {
    fs::write(path_str, genome_str).or_else(|e| Err(e.to_string()))
}

#[tauri::command]
fn load_genome(path_str: &str) -> Result<String, String> {
    fs::read_to_string(path_str).or_else(|e| Err(e.to_string()))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![save_genome, load_genome])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
