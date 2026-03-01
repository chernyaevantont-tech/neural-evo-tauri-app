use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use rfd::AsyncFileDialog;

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use entities::{DynamicBatch, DynamicTensor, GraphModel, train_simple};

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
    } else {
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
    } else {
        Ok("".to_string())
    }
}

#[tauri::command]
async fn pick_folder() -> Result<String, String> {
    let folder = AsyncFileDialog::new()
        .set_title("Select Dataset Directory")
        .pick_folder()
        .await;

    if let Some(folder_handle) = folder {
        Ok(folder_handle.path().to_string_lossy().to_string())
    } else {
        Ok("".to_string())
    }
}

#[tauri::command]
async fn test_neural_net_training(genome_str: String) -> Result<(), String> {
    type Backend = Autodiff<Wgpu>;
    let device = Default::default();

    println!("Building model from genome...");
    let model = GraphModel::<Backend>::build(&genome_str, &device);

    println!("Generating random inputs...");
    let mut inputs = Vec::new();
    for shape in &model.input_shapes {
        let tensor = if shape.len() == 1 {
            DynamicTensor::Dim2(Tensor::<Backend, 2>::random(
                [1, shape[0]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else if shape.len() == 3 {
            // shape comes as [H, W, C] - need [Batch, C, H, W]
            DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                [1, shape[2], shape[0], shape[1]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else {
            return Err("Unsupported input shape".to_string());
        };
        inputs.push(tensor);
    }

    println!("Generating random targets...");
    let mut targets = Vec::new();
    for shape in &model.output_shapes {
        let tensor = if shape.len() == 1 {
            // For Dim2, compute_loss assumes CrossEntropyLoss and expects targets to be class indices of shape [batch_size, 1].
            // num_classes is shape[0]. Class indices must be between 0 and num_classes - 1.
            let num_classes = if shape[0] > 0 { shape[0] as f64 } else { 1.0 };
            // Generate a random class using burn's Distribution::Uniform and cast it
            let random_class_tensor =
                Tensor::<Backend, 1>::random([1], Distribution::Uniform(0.0, num_classes), &device)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
            let class_idx = random_class_tensor.floor();
            let target_tensor = Tensor::<Backend, 2>::from_data([[class_idx]], &device);
            DynamicTensor::Dim2(target_tensor)
        } else if shape.len() == 3 {
            DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                [1, shape[2], shape[0], shape[1]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else {
            return Err("Unsupported output shape".to_string());
        };
        targets.push(tensor);
    }

    let batch = DynamicBatch { inputs, targets };

    println!("Starting simple training for 3 epochs...");
    train_simple(model, &[batch], &[], &[], 3, 0.005);

    println!("Training step completed successfully.");
    Ok(())
}

use image::imageops::FilterType;

#[tauri::command]
async fn test_train_on_image_folder(genome_str: String) -> Result<(), String> {
    println!("Asking user to pick a dataset folder containing 'Cat' and 'Dog' folders...");
    let folder = AsyncFileDialog::new()
        .set_title("Select 'cats_vs_dogs' root folder (with 'Cat' and 'Dog' subfolders)")
        .pick_folder()
        .await;

    let folder_path = match folder {
        Some(f) => f.path().to_path_buf(),
        None => return Err("Folder picking cancelled".to_string()),
    };

    type Backend = Autodiff<Wgpu>;
    let device = Default::default();

    println!("Building model from genome...");
    let model = GraphModel::<Backend>::build(&genome_str, &device);

    if model.input_shapes.is_empty() {
        return Err("Model has no inputs!".to_string());
    }

    let input_shape = &model.input_shapes[0];
    if input_shape.len() != 3 {
        return Err("For images, input shape must be 3-dimensional [H, W, C]".to_string());
    }

    // Read HWC from frontend schema
    let h = input_shape[0];
    let w = input_shape[1];
    let c = input_shape[2];

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    let mut load_class = |class_name: &str, label: f32, max_images: usize| {
        let class_dir = folder_path.join(class_name);
        if !class_dir.exists() {
            println!("Warning: Directory {:?} not found", class_dir);
            return;
        }

        let mut count = 0;
        if let Ok(entries) = fs::read_dir(class_dir) {
            for entry in entries.flatten() {
                if count >= max_images {
                    break;
                }

                let path = entry.path();
                if path
                    .extension()
                    .map_or(false, |ext| ext == "jpg" || ext == "jpeg" || ext == "png")
                {
                    if let Ok(img) = image::open(&path) {
                        let img = img.resize_exact(w as u32, h as u32, FilterType::Triangle);
                        let img_rgb = img.to_rgb8();

                        let mut pixels = Vec::new();
                        // For burn, NCHW means layout should be C, then H, then W
                        for c_idx in 0..c {
                            for y in 0..h {
                                for x in 0..w {
                                    let pixel = img_rgb.get_pixel(x as u32, y as u32);
                                    let val = pixel[c_idx] as f32 / 255.0;
                                    pixels.push(val);
                                }
                            }
                        }

                        let img_tensor_1d =
                            Tensor::<Backend, 1>::from_floats(pixels.as_slice(), &device);

                        let img_tensor_4d = img_tensor_1d.reshape([1, c, h, w]);

                        inputs.push(img_tensor_4d);
                        targets.push(Tensor::<Backend, 2>::from_data([[label]], &device));
                        count += 1;
                    } else {
                        println!("Failed to decode image: {:?}", path);
                    }
                }
            }
        }
        println!("Loaded {} images for class {}", count, class_name);
    };

    println!("Loading small subset of images for testing (100 per class)...");
    load_class("Cat", 0.0, 100);
    load_class("Dog", 1.0, 100);

    if inputs.is_empty() {
        return Err(
            "No images loaded! Make sure the folder contains 'Cat' and 'Dog' subfolders"
                .to_string(),
        );
    }

    // Zip inputs and targets together and shuffle
    use rand::seq::SliceRandom;
    let mut dataset: Vec<_> = inputs.into_iter().zip(targets.into_iter()).collect(); // fixed targets
    let mut rng = rand::rng();
    dataset.shuffle(&mut rng);

    // Calculate splits (70% Train, 15% Valid, 15% Test)
    let total = dataset.len();
    let train_count = (total as f32 * 0.7) as usize;
    let valid_count = (total as f32 * 0.15) as usize;

    let test_data = dataset.split_off(train_count + valid_count);
    let valid_data = dataset.split_off(train_count);
    let train_data = dataset;

    let create_batch =
        |data: Vec<(Tensor<Backend, 4>, Tensor<Backend, 2>)>| -> Option<DynamicBatch<Backend>> {
            if data.is_empty() {
                return None;
            }
            let (inp, tar): (Vec<_>, Vec<_>) = data.into_iter().unzip();
            Some(DynamicBatch {
                inputs: vec![DynamicTensor::Dim4(Tensor::cat(inp, 0))],
                targets: vec![DynamicTensor::Dim2(Tensor::cat(tar, 0))],
            })
        };

    let train_batches = create_batch(train_data)
        .map(|b| vec![b])
        .unwrap_or_default();
    let valid_batches = create_batch(valid_data)
        .map(|b| vec![b])
        .unwrap_or_default();
    let test_batches = create_batch(test_data).map(|b| vec![b]).unwrap_or_default();

    let get_batch_size = |batch: Option<&DynamicBatch<Backend>>| -> usize {
        batch
            .map(|b| match &b.inputs[0] {
                DynamicTensor::Dim2(t) => t.dims()[0],
                DynamicTensor::Dim4(t) => t.dims()[0],
            })
            .unwrap_or(0)
    };

    println!(
        "Dataset split: Train: {}, Valid: {}, Test: {}",
        get_batch_size(train_batches.first()),
        get_batch_size(valid_batches.first()),
        get_batch_size(test_batches.first())
    );

    println!("Starting simple training for 100 epochs...");
    train_simple(
        model,
        &train_batches,
        &valid_batches,
        &test_batches,
        100,
        0.0005,
    );

    println!("Image training step completed successfully.");
    Ok(())
}

#[derive(serde::Serialize)]
pub struct EvaluationResult {
    pub genome_id: String,
    pub loss: f32,
    pub accuracy: f32,
}

#[tauri::command]
async fn evaluate_population(
    genomes: Vec<String>,
    dataset_profile: String,
) -> Result<Vec<EvaluationResult>, String> {
    // Phase 2 Step 2: Parse genomes and run short training epochs
    // For now we will just successfully parse the genomes and return dummy fitness.
    type Backend = Autodiff<Wgpu>;
    let device = Default::default();

    let mut results = Vec::new();

    println!(
        "Evaluating population of {} genomes on dataset: {}",
        genomes.len(),
        dataset_profile
    );

    for (i, genome_str) in genomes.iter().enumerate() {
        // We will parse the ID out manually or assume they are passed with IDs.
        // For simplicity now, let's just parse the GraphModel to ensure it builds
        match std::panic::catch_unwind(|| GraphModel::<Backend>::build(genome_str, &device)) {
            Ok(_model) => {
                // Return dummy fitness scores simulating a quick evaluation pass
                results.push(EvaluationResult {
                    genome_id: format!("genome_{}", i),
                    loss: 0.5 + (i as f32 * 0.1),
                    accuracy: 0.8 - (i as f32 * 0.05),
                });
            }
            Err(_) => {
                println!("Failed to build model for genome {}", i);
                results.push(EvaluationResult {
                    genome_id: format!("genome_{}", i),
                    loss: 999.0,
                    accuracy: 0.0,
                });
            }
        }
    }

    Ok(results)
}

// --- Scan Dataset ---

#[derive(serde::Deserialize)]
pub struct StreamLocatorConfig {
    pub stream_id: String,
    pub alias: String,
    pub locator_type: String, // "GlobPattern" | "FolderMapping" | "CompanionFile" | "None"
    pub pattern: Option<String>, // for GlobPattern
    pub path_template: Option<String>, // for CompanionFile
}

#[derive(serde::Serialize)]
pub struct StreamScanReport {
    pub stream_id: String,
    pub alias: String,
    pub found_count: usize,
    pub missing_sample_ids: Vec<String>,
    pub discovered_classes: Option<HashMap<String, usize>>, // class_name -> count
}

#[derive(serde::Serialize)]
pub struct ScanDatasetResult {
    pub total_matched: usize,
    pub dropped_count: usize,
    pub stream_reports: Vec<StreamScanReport>,
}

fn collect_glob_ids(root: &Path, pattern: &str) -> HashMap<String, std::path::PathBuf> {
    let full_pattern = root.join(pattern).to_string_lossy().to_string();
    let full_pattern = full_pattern.replace('\\', "/");
    let mut map = HashMap::new();
    if let Ok(paths) = glob::glob(&full_pattern) {
        for entry in paths.flatten() {
            if entry.is_file() {
                // Use relative path from root (without extension) as SampleID
                // e.g. root=D:\PetImages, file=D:\PetImages\Cat\123.jpg => id="Cat/123"
                if let Ok(relative) = entry.strip_prefix(root) {
                    let id = relative
                        .with_extension("")
                        .to_string_lossy()
                        .replace('\\', "/");
                    map.insert(id, entry);
                }
            }
        }
    }
    map
}

fn collect_folder_mapping_ids(
    anchor_ids: &HashMap<String, std::path::PathBuf>,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (id, path) in anchor_ids {
        if let Some(parent) = path.parent() {
            if let Some(folder_name) = parent.file_name() {
                map.insert(id.clone(), folder_name.to_string_lossy().to_string());
            }
        }
    }
    map
}

fn collect_companion_ids(
    root: &Path,
    anchor_ids: &HashSet<String>,
    path_template: &str,
) -> HashSet<String> {
    let mut found = HashSet::new();
    for id in anchor_ids {
        let relative = path_template.replace("{id}", id);
        let full_path = root.join(&relative);
        if full_path.exists() {
            found.insert(id.clone());
        }
    }
    found
}

#[tauri::command]
async fn scan_dataset(
    root_path: String,
    stream_configs: Vec<StreamLocatorConfig>,
) -> Result<ScanDatasetResult, String> {
    let root = Path::new(&root_path);
    if !root.exists() {
        return Err(format!("Root path does not exist: {}", root_path));
    }

    // Step 1: Build the anchor — a HashMap<SampleID, PathBuf> from the first
    // GlobPattern OR FolderMapping stream. FolderMapping auto-globs for common
    // image extensions so it works standalone (ImageNet-style datasets).
    let mut anchor_ids: HashMap<String, std::path::PathBuf> = HashMap::new();

    for cfg in &stream_configs {
        match cfg.locator_type.as_str() {
            "GlobPattern" => {
                let pattern = cfg.pattern.as_deref().unwrap_or("**/*.jpg");
                anchor_ids = collect_glob_ids(root, pattern);
                break;
            }
            "FolderMapping" => {
                // Auto-glob for common image/data files in subfolders
                let extensions = ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "csv", "txt"];
                for ext in &extensions {
                    let pattern = format!("**/*.{}", ext);
                    let found = collect_glob_ids(root, &pattern);
                    for (id, path) in found {
                        anchor_ids.entry(id).or_insert(path);
                    }
                }
                if !anchor_ids.is_empty() {
                    break;
                }
            }
            _ => {}
        }
    }

    if anchor_ids.is_empty() {
        return Err(
            "No files found. Check your root directory and stream locator settings.".to_string(),
        );
    }

    let all_sample_ids: HashSet<String> = anchor_ids.keys().cloned().collect();
    let mut valid_ids = all_sample_ids.clone();

    let mut reports = Vec::new();

    for cfg in &stream_configs {
        match cfg.locator_type.as_str() {
            "GlobPattern" => {
                let pattern = cfg.pattern.as_deref().unwrap_or("*.jpg");
                let ids = collect_glob_ids(root, pattern);
                let found_set: HashSet<String> = ids.keys().cloned().collect();
                let missing: Vec<String> = all_sample_ids.difference(&found_set).cloned().collect();
                valid_ids = valid_ids.intersection(&found_set).cloned().collect();
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: found_set.len(),
                    missing_sample_ids: missing,
                    discovered_classes: None,
                });
            }
            "FolderMapping" => {
                let folder_map = collect_folder_mapping_ids(&anchor_ids);
                // Count samples per class
                let mut class_counts: HashMap<String, usize> = HashMap::new();
                for class_name in folder_map.values() {
                    *class_counts.entry(class_name.clone()).or_insert(0) += 1;
                }
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: folder_map.len(),
                    missing_sample_ids: vec![],
                    discovered_classes: Some(class_counts),
                });
            }
            "CompanionFile" => {
                let template = cfg.path_template.as_deref().unwrap_or("{id}.txt");
                let found = collect_companion_ids(root, &all_sample_ids, template);
                let missing: Vec<String> = all_sample_ids.difference(&found).cloned().collect();
                valid_ids = valid_ids.intersection(&found).cloned().collect();
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: found.len(),
                    missing_sample_ids: missing,
                    discovered_classes: None,
                });
            }
            _ => {
                // "None" or unknown - skip, don't filter
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: 0,
                    missing_sample_ids: vec![],
                    discovered_classes: None,
                });
            }
        }
    }

    let total_matched = valid_ids.len();
    let dropped_count = all_sample_ids.len().saturating_sub(total_matched);

    Ok(ScanDatasetResult {
        total_matched,
        dropped_count,
        stream_reports: reports,
    })
}

// --- Genome Library ---

fn get_genomes_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    exe_dir.join("genomes")
}

fn get_meta_path() -> PathBuf {
    get_genomes_dir().join("meta.json")
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct GenomeLibraryEntry {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub input_dims: Vec<usize>, // dimensionality of each input, e.g. [3] for one 3D image
    pub output_dims: Vec<usize>, // dimensionality of each output, e.g. [1] for one 1D vector
    pub total_nodes: usize,
    pub layer_types: Vec<String>,
    pub best_loss: Option<f32>,
    pub best_accuracy: Option<f32>,
}

fn read_meta() -> Vec<GenomeLibraryEntry> {
    let path = get_meta_path();
    if path.exists() {
        let data = fs::read_to_string(&path).unwrap_or_default();
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        vec![]
    }
}

fn write_meta(entries: &[GenomeLibraryEntry]) -> Result<(), String> {
    let dir = get_genomes_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(entries).map_err(|e| e.to_string())?;
    fs::write(get_meta_path(), json).map_err(|e| e.to_string())
}

/// Parse a serialized genome string to extract structural metadata.
/// Format: lines of JSON nodes, then "CONNECTIONS" separator, then edges
fn extract_genome_metadata(genome_str: &str) -> (Vec<usize>, Vec<usize>, usize, Vec<String>) {
    let mut input_dims = Vec::new();
    let mut output_dims = Vec::new();
    let mut total_nodes = 0usize;
    let mut layer_types_set = HashSet::new();

    for line in genome_str.lines() {
        if line == "CONNECTIONS" {
            break;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
            total_nodes += 1;
            if let Some(node_type) = obj["node"].as_str() {
                layer_types_set.insert(node_type.to_string());
                match node_type {
                    "Input" => {
                        if let Some(shape) = obj["params"]["output_shape"].as_array() {
                            input_dims.push(shape.len());
                        }
                    }
                    "Output" => {
                        if let Some(shape) = obj["params"]["input_shape"].as_array() {
                            output_dims.push(shape.len());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut layer_types: Vec<String> = layer_types_set.into_iter().collect();
    layer_types.sort();
    (input_dims, output_dims, total_nodes, layer_types)
}

#[tauri::command]
async fn list_library_genomes() -> Result<Vec<GenomeLibraryEntry>, String> {
    Ok(read_meta())
}

#[tauri::command]
async fn save_to_library(
    genome_str: String,
    name: String,
    tags: Vec<String>,
) -> Result<GenomeLibraryEntry, String> {
    let id = uuid::Uuid::new_v4().to_string();
    let dir = get_genomes_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    // Save .evog file
    let file_path = dir.join(format!("{}.evog", id));
    fs::write(&file_path, &genome_str).map_err(|e| e.to_string())?;

    // Extract metadata from genome
    let (input_dims, output_dims, total_nodes, layer_types) = extract_genome_metadata(&genome_str);

    let entry = GenomeLibraryEntry {
        id: id.clone(),
        name,
        tags,
        created_at: chrono::Utc::now().to_rfc3339(),
        input_dims,
        output_dims,
        total_nodes,
        layer_types,
        best_loss: None,
        best_accuracy: None,
    };

    // Update meta.json
    let mut meta = read_meta();
    meta.push(entry.clone());
    write_meta(&meta)?;

    Ok(entry)
}

#[tauri::command]
async fn delete_from_library(id: String) -> Result<(), String> {
    let dir = get_genomes_dir();
    let file_path = dir.join(format!("{}.evog", id));
    if file_path.exists() {
        fs::remove_file(&file_path).map_err(|e| e.to_string())?;
    }

    let mut meta = read_meta();
    meta.retain(|e| e.id != id);
    write_meta(&meta)?;

    Ok(())
}

#[tauri::command]
async fn load_library_genome(id: String) -> Result<String, String> {
    let dir = get_genomes_dir();
    let file_path = dir.join(format!("{}.evog", id));
    fs::read_to_string(&file_path).map_err(|e| format!("Failed to load genome {}: {}", id, e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            save_genome,
            load_genome,
            pick_folder,
            test_neural_net_training,
            test_train_on_image_folder,
            evaluate_population,
            scan_dataset,
            list_library_genomes,
            save_to_library,
            delete_from_library,
            load_library_genome
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
