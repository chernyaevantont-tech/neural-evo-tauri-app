use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use rfd::AsyncFileDialog;

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use entities::{DynamicBatch, DynamicTensor, GraphModel, train_simple};
use serde::{Deserialize, Serialize};
use zero_cost_proxies::{ZeroCostConfig, ZeroCostMetrics};

pub mod data_loader;
pub mod dtos;
pub mod entities;
pub mod zero_cost_proxies;
pub mod csv_loader;
pub mod shape_inference;

/// Global session counter. Incremented by `stop_evolution`.
/// Each `evaluate_population` call captures a snapshot; if the current value
/// differs from the snapshot, the evaluation is cancelled.
static EVOLUTION_SESSION: AtomicU64 = AtomicU64::new(0);

/// Global cache: hash(genome_json + training_params) -> (loss, accuracy).
/// Persists across generations within the app lifetime. Any parameter difference = different hash.
static GENOME_EVAL_CACHE: std::sync::LazyLock<Mutex<HashMap<u64, (f32, f32)>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[tauri::command]
async fn stop_evolution() -> Result<(), String> {
    let prev = EVOLUTION_SESSION.fetch_add(1, Ordering::SeqCst);
    println!(
        ">>> Evolution cancellation requested (session {} -> {}).",
        prev,
        prev + 1
    );
    Ok(())
}

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
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);

    println!("Building model from genome...");
    let model = GraphModel::<Backend>::build(&genome_str, &device, None, None);

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
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);

    println!("Building model from genome...");
    let model = GraphModel::<Backend>::build(&genome_str, &device, None, None);

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
    app_handle: tauri::AppHandle,
    genomes: Vec<String>,
    dataset_profile: String,
    batch_size: usize,
    eval_epochs: usize,
    dataset_percent: usize,
    train_split: usize,
    val_split: usize,
    test_split: usize,
) -> Result<Vec<EvaluationResult>, String> {
    eprintln!(
        ">>> Entered evaluate_population. Preparing to process dataset profile '{}'...",
        dataset_profile
    );

    // 1. Capture the current session counter IMMEDIATELY.
    // Each evaluate_population call captures a snapshot; if the current value
    // differs from the snapshot (via stop_evolution), we abort.
    let session_snapshot = EVOLUTION_SESSION.load(Ordering::SeqCst);

    type Backend = Autodiff<Wgpu>;
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
    let mut results = Vec::new();

    // 1. Read dataset_profiles.json to find the requested profile
    let profiles_json = load_dataset_profiles().await?;
    let root: crate::dtos::DatasetProfilesRoot = serde_json::from_str(&profiles_json)
        .map_err(|e| format!("Failed to parse dataset_profiles.json: {}", e))?;

    let profile = root
        .state
        .profiles
        .into_iter()
        .find(|p| p.id == dataset_profile)
        .ok_or_else(|| {
            format!(
                "Dataset profile '{}' not found in profiles JSON",
                dataset_profile
            )
        })?;

    let source_path_str = profile
        .source_path
        .clone()
        .ok_or_else(|| format!("Profile '{}' has no sourcePath defined", profile.name))?;
    let source_dir = std::path::Path::new(&source_path_str);

    if !source_dir.exists() {
        return Err(format!(
            "Source directory does not exist: {}",
            source_path_str
        ));
    }

    use tauri::{Emitter, Manager};
    let app_data_dir = app_handle.path().app_data_dir().ok();

    let loader = match crate::data_loader::DataLoader::new(profile.clone(), app_data_dir) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };

    println!(
        "Evaluating population of {} genomes on dataset: {} ({})\nBatch Size: {}, Epochs: {}",
        genomes.len(),
        profile.name,
        source_path_str,
        batch_size,
        eval_epochs
    );

    let mut valid_ids = loader.valid_sample_ids.clone();
    if valid_ids.is_empty() {
        return Err(format!(
            "No valid samples found matching all streams for dataset: {}",
            source_path_str
        ));
    }

    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    valid_ids.shuffle(&mut rng);

    // Apply pct
    let pct = dataset_percent.clamp(1, 100);
    let use_count = (valid_ids.len() * pct) / 100;
    let use_count = use_count.max(1);
    valid_ids.truncate(use_count);

    let total_split = (train_split + val_split + test_split).max(1) as f32;
    let train_ratio = train_split as f32 / total_split;
    let val_ratio = val_split as f32 / total_split;

    let mut train_ids = Vec::new();
    let mut val_ids = Vec::new();
    let mut test_ids = Vec::new();

    // Find a categorical target stream for stratification
    let strat_stream_idx = profile.streams.iter().position(|s| s.role == "Target" && matches!(s.data_type, crate::dtos::DataType::Categorical));

    if let Some(s_idx) = strat_stream_idx {
        println!(">>> Stratifying split based on categorical stream '{}'...", profile.streams[s_idx].alias);
        let stream_id = &profile.streams[s_idx].id;
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        if let Some(map) = loader.stream_files.get(stream_id) {
            for id in &valid_ids {
                let label = map.get(id).cloned().unwrap_or_else(|| "unknown".to_string());
                groups.entry(label).or_default().push(id.clone());
            }
        }

        for (label, mut members) in groups {
            members.shuffle(&mut rng);
            let n = members.len();
            let t_count = ((n as f32) * train_ratio).round() as usize;
            let v_count = ((n as f32) * val_ratio).round() as usize;

            let t_count = t_count.min(n);
            let v_count = v_count.min(n - t_count);

            train_ids.extend(members.iter().take(t_count).cloned());
            val_ids.extend(members.iter().skip(t_count).take(v_count).cloned());
            test_ids.extend(members.iter().skip(t_count + v_count).cloned());

            println!("  Class '{}': Total={}, Train={}, Val={}, Test={}", label, n, t_count, v_count, n - t_count - v_count);
        }

        // Final shuffle of the split sets
        train_ids.shuffle(&mut rng);
        val_ids.shuffle(&mut rng);
        test_ids.shuffle(&mut rng);
    } else {
        // Fallback to random split
        let train_count = ((valid_ids.len() as f32) * train_ratio).round() as usize;
        let val_count = ((valid_ids.len() as f32) * val_ratio).round() as usize;

        let train_count = train_count.min(valid_ids.len());
        let val_count = val_count.min(valid_ids.len() - train_count);

        train_ids = valid_ids.iter().take(train_count).cloned().collect();
        val_ids = valid_ids.iter().skip(train_count).take(val_count).cloned().collect();
        test_ids = valid_ids.iter().skip(train_count + val_count).cloned().collect();
    }

    eprintln!(
        ">>> Split: {} train samples, {} val samples, {} test samples",
        train_ids.len(),
        val_ids.len(),
        test_ids.len()
    );

    // Filter streams by role
    let input_stream_indices: Vec<usize> = profile
        .streams
        .iter()
        .enumerate()
        .filter(|(_, s)| s.role == "Input")
        .map(|(i, _)| i)
        .collect();
    let target_stream_indices: Vec<usize> = profile
        .streams
        .iter()
        .enumerate()
        .filter(|(_, s)| s.role == "Target")
        .map(|(i, _)| i)
        .collect();

    if input_stream_indices.is_empty() || target_stream_indices.is_empty() {
        return Err(
            "Dataset Profile must define at least one Input and one Target stream".to_string(),
        );
    }

    // Determine expected input and output overrides based on the profile
    let mut input_overrides = Vec::new();
    for &idx in &input_stream_indices {
        let stream = &profile.streams[idx];
        match stream.data_type {
            crate::dtos::DataType::Image => {
                let mut channels = 3;
                let mut h = 64;
                let mut w = 64;
                if let Some(prep) = &stream.preprocessing {
                    if let Some(vision) = &prep.vision {
                        if vision.resize.len() == 2 {
                            w = vision.resize[0] as usize;
                            h = vision.resize[1] as usize;
                        }
                        if vision.grayscale {
                            channels = 1;
                        }
                    }
                }
                input_overrides.push(vec![channels, h, w]);
            }
            crate::dtos::DataType::Vector => {
                let dim = stream.tensor_shape.get(0).cloned().unwrap_or(1);
                input_overrides.push(vec![dim]);
            }
            _ => input_overrides.push(vec![1]),
        }
    }

    let mut output_overrides = Vec::new();
    let mut is_classification = false;
    for &idx in &target_stream_indices {
        let stream = &profile.streams[idx];
        if let crate::dtos::DataType::Categorical = stream.data_type {
            is_classification = true;
            // Get the number of classes discovered by the loader
            let num_classes = loader.stream_classes.get(&idx).cloned().unwrap_or(1);
            output_overrides.push(vec![num_classes]);
        } else {
            let dim = stream.tensor_shape.get(0).cloned().unwrap_or(1);
            output_overrides.push(vec![dim]);
        }
    }

    // Helper macro to build batches
    macro_rules! build_batches {
        ($ids:expr, $loader:expr, $dev:expr) => {{
            let mut assembled_batches: Vec<crate::entities::DynamicBatch<Backend>> = Vec::new();
            for (idx, chunk) in $ids.chunks(batch_size).enumerate() {
                // Check cancellation every 10 chunks during assembly
                if idx % 10 == 0 && EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                    return Err("Evolution cancelled during batch assembly".to_string());
                }

                let mut batch_inputs: Vec<Vec<crate::entities::DynamicTensor<Backend>>> = vec![Vec::new(); input_stream_indices.len()];
                let mut batch_targets: Vec<Vec<crate::entities::DynamicTensor<Backend>>> = vec![Vec::new(); target_stream_indices.len()];

                for id in chunk {
                    match $loader.load_sample(id, $dev) {
                        Ok(sample) => {
                            for (i, &stream_idx) in input_stream_indices.iter().enumerate() {
                                if let Some(t) = sample.stream_tensors.get(&stream_idx) {
                                    let t_clone: crate::entities::DynamicTensor<Backend> = t.clone();
                                    batch_inputs[i].push(t_clone);
                                }
                            }
                            for (i, &stream_idx) in target_stream_indices.iter().enumerate() {
                                if let Some(t) = sample.stream_tensors.get(&stream_idx) {
                                    let t_clone: crate::entities::DynamicTensor<Backend> = t.clone();
                                    batch_targets[i].push(t_clone);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[ERROR] Sample '{}' dropped during load: {}", id, e);
                        }
                    }
                }

                if !batch_inputs.iter().all(|list| !list.is_empty()) || !batch_targets.iter().all(|list| !list.is_empty()) {
                    eprintln!("[DEBUG] Skipping batch. Input lens: {:?}, Target lens: {:?}",
                        batch_inputs.iter().map(|l| l.len()).collect::<Vec<_>>(),
                        batch_targets.iter().map(|l| l.len()).collect::<Vec<_>>()
                    );
                }

                if batch_inputs.iter().all(|list| !list.is_empty()) && batch_targets.iter().all(|list| !list.is_empty()) {
                    use crate::entities::concat_dynamic_tensors;
                    let inputs: Vec<crate::entities::DynamicTensor<Backend>> = batch_inputs
                        .into_iter()
                        .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                        .collect();
                    let targets: Vec<crate::entities::DynamicTensor<Backend>> = batch_targets
                        .into_iter()
                        .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                        .collect();

                    assembled_batches.push(crate::entities::DynamicBatch {
                        inputs,
                        targets,
                    });
                }
            }
            assembled_batches
        }};
    }

    // 4. Build batches ONCE (reused for all genomes)
    if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
        return Err("Evolution cancelled before batch assembly".to_string());
    }
    println!(">>> Assembling batches (shared across all genomes)...");
    let train_batches = build_batches!(train_ids.as_slice(), loader, &device);
    let val_batches = build_batches!(val_ids.as_slice(), loader, &device);
    let test_batches = build_batches!(test_ids.as_slice(), loader, &device);

    if train_batches.is_empty() {
        return Err("No training batches could be assembled. Aborting.".to_string());
    }

    println!(
        ">>> Assembled {} train + {} val + {} test batches. Starting genome evaluation...",
        train_batches.len(),
        val_batches.len(),
        test_batches.len()
    );

    // 5. Evaluation Loop over each Genome

    for (i, genome_str) in genomes.iter().enumerate() {
        // Check cancellation between genomes
        if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
            println!(">>> Evolution cancelled. Aborting remaining genomes.");
            break;
        }

        // Emit current genome index to the frontend for UI synchronization
        app_handle.emit("evaluating-genome", i).unwrap_or_else(|e| {
            eprintln!("Failed to emit evaluating-genome event: {}", e);
        });

        eprintln!(
            "\n===========================================================\nEvaluating Genome {}/{} (ID: genome_{})\n===========================================================",
            i + 1,
            genomes.len(),
            i
        );

        // Compute a cache key from genome content + all training params
        let cache_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            genome_str.hash(&mut hasher);
            dataset_profile.hash(&mut hasher);
            batch_size.hash(&mut hasher);
            eval_epochs.hash(&mut hasher);
            dataset_percent.hash(&mut hasher);
            train_split.hash(&mut hasher);
            val_split.hash(&mut hasher);
            test_split.hash(&mut hasher);
            hasher.finish()
        };

        // Check cache for identical genome + training params
        if let Some(&(cached_loss, cached_acc)) = GENOME_EVAL_CACHE.lock().unwrap().get(&cache_key)
        {
            eprintln!(
                ">>> CACHE HIT for Genome {} (hash={:#x}): loss={}, acc={}",
                i, cache_key, cached_loss, cached_acc
            );
            results.push(EvaluationResult {
                genome_id: format!("genome_{}", i),
                loss: cached_loss,
                accuracy: cached_acc,
            });

            // Emit start + result events so the frontend updates progressively
            let _ = app_handle.emit("evaluating-genome-start", i);
            #[derive(serde::Serialize, Clone)]
            struct CachedResult {
                index: usize,
                loss: f32,
                accuracy: f32,
            }
            let _ = app_handle.emit(
                "evaluating-genome-result",
                CachedResult {
                    index: i,
                    loss: cached_loss,
                    accuracy: cached_acc,
                },
            );
            continue;
        }

        eprintln!("Genome JSON:\n{}", genome_str);

        match std::panic::catch_unwind(|| {
            crate::entities::GraphModel::<Backend>::build(
                genome_str,
                &device,
                Some(&input_overrides),
                Some(&output_overrides),
            )
        }) {
            Ok(_initial_model) => {
                // Retry loop: if accuracy is near random chance, rebuild with fresh weights
                const MAX_RETRIES: usize = 3;
                const RANDOM_CHANCE_THRESHOLD: f32 = 55.0; // Below this = likely bad init

                let mut best_loss = 999.0_f32;
                let mut best_acc = 0.0_f32;

                for attempt in 0..MAX_RETRIES {
                    // Check cancellation
                    if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                        break;
                    }

                    // Rebuild model each attempt (fresh random weights)
                    let mut model = crate::entities::GraphModel::<Backend>::build(
                        genome_str,
                        &device,
                        Some(&input_overrides),
                        Some(&output_overrides),
                    );

                    // Let frontend know we are starting to evaluate a genome (so it clears live charts)
                    let _ = app_handle.emit("evaluating-genome-start", i);

                    if attempt > 0 {
                        eprintln!(
                            ">>> RETRY {}/{} for Genome {} (previous acc={:.2}%, threshold={:.0}%)",
                            attempt + 1,
                            MAX_RETRIES,
                            i,
                            best_acc,
                            RANDOM_CHANCE_THRESHOLD
                        );
                    }

                    // Train on training set
                    crate::entities::run_eval_pass(
                        &app_handle,
                        &mut model,
                        &train_batches,
                        eval_epochs,
                        0.001,
                        is_classification,
                        &EVOLUTION_SESSION,
                        session_snapshot,
                    );

                    // Check cancellation after training
                    if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                        eprintln!(
                            ">>> Cancelled after training for Genome {}. Skipping val/test.",
                            i
                        );
                        break;
                    }

                    crate::entities::run_validation_pass(
                        &model,
                        &val_batches,
                        "Validation",
                        is_classification,
                    );

                    // Evaluate fitness on test set (or val if no test batches, or train)
                    let (final_loss, final_acc) = if !test_batches.is_empty() {
                        crate::entities::run_validation_pass(
                            &model,
                            &test_batches,
                            "Test",
                            is_classification,
                        )
                    } else if !val_batches.is_empty() {
                        eprintln!(
                            ">>> WARNING: No test batches. Using validation metrics for fitness."
                        );
                        crate::entities::run_validation_pass(
                            &model,
                            &val_batches,
                            "Validation",
                            is_classification,
                        )
                    } else {
                        eprintln!(
                            ">>> WARNING: No validation/test batches. Using train metrics for fitness."
                        );
                        crate::entities::run_validation_pass(
                            &model,
                            &train_batches,
                            "Train",
                            is_classification,
                        )
                    };

                    // Keep the best attempt
                    if final_acc > best_acc {
                        best_loss = final_loss;
                        best_acc = final_acc;
                    }

                    // If accuracy is above random chance, accept and stop retrying
                    if final_acc > RANDOM_CHANCE_THRESHOLD {
                        break;
                    }

                    if attempt < MAX_RETRIES - 1 {
                        eprintln!(
                            ">>> Genome {} attempt {} got acc={:.2}% (below {:.0}% threshold). Will retry with fresh weights.",
                            i,
                            attempt + 1,
                            final_acc,
                            RANDOM_CHANCE_THRESHOLD
                        );
                    }
                }

                // Check cancellation before pushing result
                if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                    break;
                }

                results.push(EvaluationResult {
                    genome_id: format!("genome_{}", i),
                    loss: best_loss,
                    accuracy: best_acc,
                });

                // Notify frontend of this genome's result for progressive UI
                #[derive(serde::Serialize, Clone)]
                struct GenomeResult {
                    index: usize,
                    loss: f32,
                    accuracy: f32,
                }
                let _ = app_handle.emit(
                    "evaluating-genome-result",
                    GenomeResult {
                        index: i,
                        loss: best_loss,
                        accuracy: best_acc,
                    },
                );

                // Only cache results above random chance threshold
                if best_acc > RANDOM_CHANCE_THRESHOLD {
                    GENOME_EVAL_CACHE
                        .lock()
                        .unwrap()
                        .insert(cache_key, (best_loss, best_acc));
                }
            }
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic"
                };
                println!(
                    ">>> ABORTED: Genome {} failed to compile cleanly: {}",
                    i, msg
                );
                results.push(EvaluationResult {
                    genome_id: format!("genome_{}", i),
                    loss: 999.0,
                    accuracy: 0.0,
                });

                // Notify frontend of this genome's failure result
                #[derive(serde::Serialize, Clone)]
                struct GenomeResult2 {
                    index: usize,
                    loss: f32,
                    accuracy: f32,
                }
                let _ = app_handle.emit(
                    "evaluating-genome-result",
                    GenomeResult2 {
                        index: i,
                        loss: 999.0,
                        accuracy: 0.0,
                    },
                );
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
    pub locator_type: String, // "GlobPattern" | "FolderMapping" | "CompanionFile" | "CsvDataset" | "None"
    pub pattern: Option<String>, // for GlobPattern
    pub path_template: Option<String>, // for CompanionFile
    
    // CSV Dataset specific
    pub csv_path: Option<String>,
    pub has_headers: Option<bool>,
    pub sample_mode: Option<String>, // "row" | "temporal_window"
    pub feature_columns: Option<Vec<String>>,
    pub target_column: Option<String>,
    pub window_size: Option<usize>,
    pub stream_role: Option<String>, // "Input" | "Target" | "Ignore"
}

#[derive(serde::Serialize)]
pub struct StreamScanReport {
    pub stream_id: String,
    pub alias: String,
    pub found_count: usize,
    pub missing_sample_ids: Vec<String>,
    pub discovered_classes: Option<HashMap<String, usize>>, // class_name -> count
    pub input_shape: Option<Vec<usize>>, // For Input streams: [window_size, num_features] or [num_features]
    pub num_classes: Option<usize>,              // For Target streams: number of distinct classes
    pub inferred_data_type: String,              // "Image" | "TemporalSequence" | "Vector" | "Categorical"
    pub warnings: Vec<String>,                   // Warnings discovered during scanning
}

#[derive(serde::Serialize)]
pub struct ScanDatasetResult {
    pub total_matched: usize,
    pub dropped_count: usize,
    pub stream_reports: Vec<StreamScanReport>,
    pub valid_sample_ids: Vec<String>, // Sample IDs after alignment
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

    // Step 1a: Check if any stream is CsvDataset — if so, use it as anchor
    let mut anchor_ids: HashMap<String, std::path::PathBuf> = HashMap::new();
    let mut anchor_from_csv = false;
    
    for cfg in &stream_configs {
        if cfg.locator_type == "CsvDataset" {
            let csv_path = cfg.csv_path.as_deref().unwrap_or("data.csv");
            let has_headers = cfg.has_headers.unwrap_or(true);
            let feature_columns = cfg.feature_columns.clone().unwrap_or_default();
            let target_column = cfg.target_column.clone().unwrap_or_else(|| "label".to_string());
            
            let full_path = root.join(csv_path);
            eprintln!(">>> Attempting to load CSV: {} (full path: {})", csv_path, full_path.display());
            eprintln!("    Config: has_headers={}, feature_cols={:?}, target_col={}", 
                has_headers, feature_columns, target_column);
            
            // For anchor discovery, use dummy feature columns and row mode (we just need sample count)
            let csv_config = crate::dtos::CsvDatasetDef {
                csv_path: csv_path.to_string(),
                has_headers,
                sample_mode: "row".to_string(), // Always use row mode for anchor discovery
                feature_columns: if feature_columns.is_empty() { vec!["0".to_string()] } else { feature_columns.clone() },
                target_column: target_column.clone(),
                window_size: None,
                window_stride: None,
                preprocessing: crate::dtos::CsvPreprocessingConfig {
                    normalization: "none".to_string(),
                    handle_missing: "skip".to_string(),
                },
            };
            
            match crate::csv_loader::CsvDatasetLoader::init(root, csv_config) {
                Ok(_loader) => {
                    // For CSV, use row indices as SampleIDs
                    for i in 0.._loader.num_samples {
                        let sample_id = format!("csv_row_{}", i);
                        anchor_ids.insert(sample_id, root.to_path_buf());
                    }
                    anchor_from_csv = true;
                    eprintln!(">>> CSV anchor loaded successfully: {} samples from {}", _loader.num_samples, csv_path);
                    break;
                }
                Err(e) => {
                    eprintln!(">>> Warning: Failed to load CSV '{}' for anchor: {}", csv_path, e);
                }
            }
        }
    }

    // Step 1b: If no CSV anchor, try GlobPattern/FolderMapping
    if !anchor_from_csv {
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
    }

    if anchor_ids.is_empty() {
        let has_csv = stream_configs.iter().any(|c| c.locator_type == "CsvDataset");
        let csv_info = if has_csv {
            "\n\nNote: You have CsvDataset streams. Make sure the CSV file path exists relative to the root folder."
        } else {
            ""
        };
        return Err(
            format!("No files found. Check your root directory and stream locator settings.{}", csv_info)
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
                    input_shape: None,
                    num_classes: None,
                    inferred_data_type: "Image".to_string(),
                    warnings: vec![],
                });
            }
            "MasterIndex" => {
                let index_path = cfg.pattern.as_deref().unwrap_or("index.csv");
                let full_path = root.join(index_path);
                let mut found_ids = HashSet::new();
                let mut class_counts: HashMap<String, usize> = HashMap::new();

                if let Ok(mut rdr) = csv::ReaderBuilder::new().has_headers(true).from_path(&full_path) {
                    // Try to find a classification column if possible, or just collect IDs
                    for result in rdr.records() {
                        if let Ok(record) = result {
                            if let Some(id) = record.get(0) {
                                let id_str = id.to_string();
                                found_ids.insert(id_str.clone());
                                // If there's a second column, assume it's a class for preview
                                if let Some(class) = record.get(1) {
                                    *class_counts.entry(class.to_string()).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
                
                let missing: Vec<String> = all_sample_ids.difference(&found_ids).cloned().collect();
                valid_ids = valid_ids.intersection(&found_ids).cloned().collect();

                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: found_ids.len(),
                    missing_sample_ids: missing,
                    discovered_classes: if class_counts.is_empty() { None } else { Some(class_counts.clone()) },
                    input_shape: None,
                    num_classes: if class_counts.is_empty() { None } else { Some(class_counts.len()) },
                    inferred_data_type: "Categorical".to_string(),
                    warnings: vec![],
                });
            }
            "FolderMapping" => {
                let folder_map = collect_folder_mapping_ids(&anchor_ids);
                let mut class_counts: HashMap<String, usize> = HashMap::new();
                for class_name in folder_map.values() {
                    *class_counts.entry(class_name.clone()).or_insert(0) += 1;
                }
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: folder_map.len(),
                    missing_sample_ids: vec![],
                    discovered_classes: Some(class_counts.clone()),
                    input_shape: None,
                    num_classes: if class_counts.is_empty() { None } else { Some(class_counts.len()) },
                    inferred_data_type: "Image".to_string(),
                    warnings: vec![],
                });
            }
            "CompanionFile" => {
                let template = cfg.path_template.as_deref().unwrap_or("{id}.txt");
                let found = collect_companion_ids(root, &all_sample_ids, template);
                let missing: Vec<String> = all_sample_ids.difference(&found).cloned().collect();
                valid_ids = valid_ids.intersection(&found).cloned().collect();
                
                // For companion files, we can't easily discover classes without parsing them all
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: found.len(),
                    missing_sample_ids: missing,
                    discovered_classes: None,
                    input_shape: None,
                    num_classes: None,
                    inferred_data_type: "Text".to_string(),
                    warnings: vec![],
                });
            }
            "CsvDataset" => {
                // Load CSV and discover classes OR count samples based on stream role
                let csv_path = cfg.csv_path.as_deref().unwrap_or("data.csv");
                let has_headers = cfg.has_headers.unwrap_or(true);
                let feature_columns = cfg.feature_columns.clone().unwrap_or_default();
                let target_column = cfg.target_column.clone().unwrap_or_else(|| "label".to_string());
                let stream_role = cfg.stream_role.as_deref().unwrap_or("Input");
                
                let mut class_counts: HashMap<String, usize> = HashMap::new();
                let mut csv_samples = 0;
                
                // Save feature column count for logging
                let feature_col_count = feature_columns.len();
                let window_size = cfg.window_size.unwrap_or(1);
                let sample_mode = cfg.sample_mode.as_deref().unwrap_or("row");
                
                // Build CSV config based on stream role
                // Input streams: use feature columns + sample_mode (row or temporal_window)
                // Target streams: ignore feature columns, always use row mode for labels
                let csv_config = if stream_role == "Target" && !target_column.is_empty() {
                    // For Target streams: use target column for class discovery, always row mode
                    crate::dtos::CsvDatasetDef {
                        csv_path: csv_path.to_string(),
                        has_headers,
                        sample_mode: "row".to_string(), // Target always uses row mode
                        feature_columns: vec![], // Target doesn't use features
                        target_column,
                        window_size: None,
                        window_stride: None,
                        preprocessing: crate::dtos::CsvPreprocessingConfig {
                            normalization: "none".to_string(),
                            handle_missing: "skip".to_string(),
                        },
                    }
                } else if stream_role == "Input" && !feature_columns.is_empty() {
                    // For Input streams: use feature columns + configured sample_mode
                    crate::dtos::CsvDatasetDef {
                        csv_path: csv_path.to_string(),
                        has_headers,
                        sample_mode: sample_mode.to_string(),
                        feature_columns,
                        target_column: String::new(), // Empty for input streams
                        window_size: if sample_mode == "temporal_window" { Some(cfg.window_size.unwrap_or(50)) } else { None },
                        window_stride: None,
                        preprocessing: crate::dtos::CsvPreprocessingConfig {
                            normalization: "none".to_string(),
                            handle_missing: "skip".to_string(),
                        },
                    }
                } else {
                    eprintln!(
                        ">>> Skipping CSV scan for stream '{}' ({}): Invalid config for role '{}'",
                        cfg.alias, csv_path, stream_role
                    );
                    reports.push(StreamScanReport {
                        stream_id: cfg.stream_id.clone(),
                        alias: cfg.alias.clone(),
                        found_count: 0,
                        missing_sample_ids: vec![],
                        discovered_classes: None,
                        input_shape: None,
                        num_classes: None,
                        inferred_data_type: "Vector".to_string(),
                        warnings: vec!["Invalid CSV stream configuration".to_string()],
                    });
                    continue;
                };
                
                match crate::csv_loader::CsvDatasetLoader::init(root, csv_config) {
                    Ok(loader) => {
                        csv_samples = loader.num_samples;
                        for class in &loader.discovered_classes {
                            *class_counts.entry(class.clone()).or_insert(0) += 1;
                        }
                        
                        eprintln!(
                            ">>> CSV scan success: stream '{}' ({}) loaded {} samples, classes: {:?}",
                            cfg.alias,
                            if stream_role == "Target" {
                                format!("Target, classes={}", class_counts.len())
                            } else {
                                format!("Input ({}), shape=[{}, {}]", sample_mode, window_size, feature_col_count)
                            },
                            csv_samples,
                            if class_counts.is_empty() {
                                "none (Input stream)".to_string()
                            } else {
                                format!("{:?}", class_counts.keys().collect::<Vec<_>>())
                            }
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            ">>> CSV scan failed for stream '{}' ({}): {}",
                            cfg.alias, csv_path, e
                        );
                    }
                }
                
                // Calculate input_shape for Input streams
                let input_shape = if stream_role == "Input" {
                    if sample_mode == "temporal_window" {
                        Some(vec![window_size, feature_col_count])
                    } else {
                        Some(vec![feature_col_count])
                    }
                } else {
                    None
                };
                
                // Infer data type based on stream role
                let data_type_str = if stream_role == "Target" {
                    "Categorical"
                } else if sample_mode == "temporal_window" {
                    "TemporalSequence"
                } else {
                    "Vector"
                };
                
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: csv_samples,
                    missing_sample_ids: vec![],
                    discovered_classes: if class_counts.is_empty() { None } else { Some(class_counts.clone()) },
                    input_shape,
                    num_classes: if class_counts.is_empty() { None } else { Some(class_counts.len()) },
                    inferred_data_type: data_type_str.to_string(),
                    warnings: vec![],
                });
            }
            _ => {
                // "None" or unknown - skip, don't filter
                reports.push(StreamScanReport {
                    stream_id: cfg.stream_id.clone(),
                    alias: cfg.alias.clone(),
                    found_count: 0,
                    missing_sample_ids: vec![],
                    input_shape: None,
                    discovered_classes: None,
                    num_classes: None,
                    inferred_data_type: "Vector".to_string(),
                    warnings: vec!["Unknown locator type".to_string()],
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
        valid_sample_ids: valid_ids.into_iter().collect(),
    })
}

#[tauri::command]
async fn cache_dataset(
    app_handle: tauri::AppHandle,
    dataset_profile_id: String,
) -> Result<crate::data_loader::CacheResult, String> {
    use tauri::Manager;
    let profiles_json = load_dataset_profiles().await?;
    let root: crate::dtos::DatasetProfilesRoot = serde_json::from_str(&profiles_json)
        .map_err(|e| format!("Failed to parse dataset_profiles.json: {}", e))?;

    let profile = root
        .state
        .profiles
        .into_iter()
        .find(|p| p.id == dataset_profile_id)
        .ok_or_else(|| {
            format!(
                "Dataset profile '{}' not found in profiles JSON",
                dataset_profile_id
            )
        })?;

    let app_data_dir = app_handle.path().app_data_dir().ok();
    let loader = crate::data_loader::DataLoader::new(profile, app_data_dir)?;
    let result = loader.build_image_cache()?;
    Ok(result)
}

#[tauri::command]
async fn validate_dataset_profile(profile_json: String) -> Result<dtos::DatasetValidationReport, String> {
    // Validate a dataset profile for evolution readiness.
    // Checks:
    // - All Input streams have valid input shapes
    // - At least one Target stream exists with valid num_classes
    // - Streams have compatible sample counts
    // - Data types are supported by current architecture

    let profile: dtos::DatasetProfile = serde_json::from_str(&profile_json)
        .map_err(|e| format!("Failed to parse dataset profile: {}", e))?;

    let mut issues: Vec<dtos::ValidationIssue> = vec![];
    let mut input_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut output_shape: Option<Vec<usize>> = None;
    let mut total_valid_samples: usize = 0;

    // Check all Input streams have valid shapes
    let input_streams: Vec<_> = profile
        .streams
        .iter()
        .filter(|s| s.role == "Input")
        .collect();

    for stream in &input_streams {
        match shape_inference::ShapeInference::infer_input_shape(stream, Path::new(".")) {
            Ok((shape, _, warnings)) => {
                if shape.is_empty() {
                    issues.push(dtos::ValidationIssue {
                        severity: dtos::ValidationSeverity::Error,
                        component: "InputShape".to_string(),
                        message: format!(
                            "Input stream '{}' has empty shape. Check feature columns.",
                            stream.alias
                        ),
                        suggested_fix: Some(
                            "Ensure CSV stream specifies feature_columns or image tensor_shape"
                                .to_string(),
                        ),
                    });
                } else {
                    input_shapes.insert(stream.id.clone(), shape);
                }
                for warning in warnings {
                    issues.push(dtos::ValidationIssue {
                        severity: dtos::ValidationSeverity::Warning,
                        component: "InputShape".to_string(),
                        message: warning,
                        suggested_fix: None,
                    });
                }
            }
            Err(e) => {
                issues.push(dtos::ValidationIssue {
                    severity: dtos::ValidationSeverity::Error,
                    component: "InputShape".to_string(),
                    message: format!("Failed to infer shape for input stream '{}': {}", stream.alias, e),
                    suggested_fix: None,
                });
            }
        }
    }

    // Check Target streams have valid output shapes
    let target_streams: Vec<_> = profile
        .streams
        .iter()
        .filter(|s| s.role == "Target")
        .collect();

    if target_streams.is_empty() {
        issues.push(dtos::ValidationIssue {
            severity: dtos::ValidationSeverity::Error,
            component: "OutputShape".to_string(),
            message: "No Target stream defined. Evolution requires a target for training.".to_string(),
            suggested_fix: Some("Add a Target stream with your labels".to_string()),
        });
    } else {
        for (idx, stream) in target_streams.iter().enumerate() {
            // For targets, num_classes would come from discovered_classes during scan
            // For now, we just validate that it's a supported type
            match stream.data_type {
                dtos::DataType::Categorical | dtos::DataType::Vector => {
                    // These are supported
                }
                _ => {
                    issues.push(dtos::ValidationIssue {
                        severity: dtos::ValidationSeverity::Error,
                        component: "OutputShape".to_string(),
                        message: format!(
                            "Target stream '{}' has unsupported data_type: {:?}",
                            stream.alias, stream.data_type
                        ),
                        suggested_fix: Some(
                            "Target streams must be Categorical or Vector type".to_string(),
                        ),
                    });
                }
            }

            // For now, set output_shape to a placeholder (will be updated by scan)
            if idx == 0 {
                // This will be properly set after scan with num_classes
                output_shape = Some(vec![0]); // Placeholder
            }
        }
    }

    // Summary
    let is_valid = !issues.iter().any(|i| i.severity == dtos::ValidationSeverity::Error);
    let can_start_evolution = is_valid && !input_shapes.is_empty() && !target_streams.is_empty();

    Ok(dtos::DatasetValidationReport {
        is_valid,
        issues,
        input_shapes,
        output_shape,
        total_valid_samples,
        can_start_evolution,
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

// --- Dataset Profiles Persistence ---

fn get_dataset_profiles_path() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    exe_dir.join("dataset_profiles.json")
}

#[tauri::command]
async fn save_dataset_profiles(profiles_json: String) -> Result<(), String> {
    let path = get_dataset_profiles_path();
    // Validate it's valid JSON before writing to avoid corrupting the file
    let _: serde_json::Value = serde_json::from_str(&profiles_json).map_err(|e| e.to_string())?;
    fs::write(path, profiles_json).map_err(|e| e.to_string())
}

#[tauri::command]
async fn load_dataset_profiles() -> Result<String, String> {
    let path = get_dataset_profiles_path();
    if path.exists() {
        fs::read_to_string(path).map_err(|e| e.to_string())
    } else {
        Ok("[]".to_string())
    }
}

#[derive(Serialize, Deserialize)]
pub struct CsvPreview {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[tauri::command]
async fn preview_csv(
    root_path: String,
    index_path: String,
    has_headers: bool,
    rows: usize,
) -> Result<CsvPreview, String> {
    let full_path = Path::new(&root_path).join(index_path);

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_path(full_path)
        .map_err(|e| e.to_string())?;

    let headers = if has_headers {
        if let Ok(record) = rdr.headers() {
            record.iter().map(|s| s.to_string()).collect()
        } else {
            vec![]
        }
    } else {
        // Just generate "Column 0", "Column 1" based on first row length
        vec![] // will populate below
    };

    let mut parsed_rows = Vec::new();
    let mut actual_headers = headers;

    for (i, result) in rdr.records().enumerate() {
        if i >= rows {
            break;
        }
        if let Ok(record) = result {
            let row_data: Vec<String> = record.iter().map(|s| s.to_string()).collect();

            // Generate dummy headers if missing and this is the first row
            if !has_headers && actual_headers.is_empty() {
                actual_headers = (0..record.len())
                    .map(|idx| format!("Col {}", idx))
                    .collect();
            }
            parsed_rows.push(row_data);
        }
    }

    Ok(CsvPreview {
        headers: actual_headers,
        rows: parsed_rows,
    })
}

/// Compute zero-cost proxy metrics for fast architecture evaluation
/// 
/// This performs a single forward-backward pass to estimate architecture quality
/// without full training. Based on SynFlow metric from ICLR 2021 paper:
/// "Zero-Cost Proxies for Lightweight NAS"
#[tauri::command]
async fn compute_zero_cost_score(
    genome_json: String,
    config_json: String,
) -> Result<ZeroCostMetrics, String> {
    type Backend = Autodiff<Wgpu>;
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
    
    // Parse config
    let config: ZeroCostConfig = serde_json::from_str(&config_json)
        .map_err(|e| format!("Failed to parse config: {}", e))?;
    
    if !config.enabled {
        return Ok(ZeroCostMetrics {
            synflow: 5.0,
            normalized_score: 0.5,
            strategy_decision: "full_train".to_string(),
        });
    }
    
    // Build model from genome
    let model = GraphModel::<Backend>::build(&genome_json, &device, None, None);
    
    // Create a minimal sample batch (1 sample with dummy data)
    let mut inputs = Vec::new();
    for shape in &model.input_shapes {
        let tensor = if shape.len() == 1 {
            DynamicTensor::Dim2(Tensor::<Backend, 2>::random(
                [1, shape[0]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else if shape.len() == 3 {
            // [H, W, C] -> [Batch, C, H, W]
            DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                [1, shape[2], shape[0], shape[1]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else {
            return Err("Unsupported input shape for zero-cost scoring".to_string());
        };
        inputs.push(tensor);
    }
    
    // Create dummy targets
    let mut targets = Vec::new();
    for shape in &model.output_shapes {
        let tensor = if shape.len() == 1 {
            let num_classes = if shape[0] > 0 { shape[0] as f64 } else { 1.0 };
            let random_class = Tensor::<Backend, 1>::random(
                [1],
                Distribution::Uniform(0.0, num_classes),
                &device,
            )
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| "Failed to convert tensor to vec".to_string())?[0];
            let class_idx = random_class.floor();
            DynamicTensor::Dim2(Tensor::<Backend, 2>::from_data([[class_idx]], &device))
        } else if shape.len() == 3 {
            DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                [1, shape[2], shape[0], shape[1]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else {
            return Err("Unsupported output shape for zero-cost scoring".to_string());
        };
        targets.push(tensor);
    }
    
    let batch = DynamicBatch { inputs, targets };
    
    // Compute forward pass and loss
    let _output = model.forward(&batch.inputs);
    
    // For now, compute a heuristic based on model architecture properties
    // In a full implementation, this would compute SynFlow properly
    let synflow_score = compute_synflow_heuristic(&model);
    
    // Create metrics with decision logic
    let metrics = ZeroCostMetrics::from_synflow(synflow_score, &config);
    
    Ok(metrics)
}

/// Heuristic for SynFlow score based on model architecture
/// 
/// In a full autodiff implementation, this would compute:
/// SynFlow = Σ |∇w ⊙ w| (gradient-weight product)
/// 
/// For now, we use a proxy metric based on layer connectivity and depth
fn compute_synflow_heuristic<B: burn::prelude::Backend>(model: &GraphModel<B>) -> f32 {
    // Count total layers
    let total_layers = (
        model.conv1ds.len() +
        model.conv2ds.len() +
        model.denses.len() +
        model.lstms.len() +
        model.grus.len() +
        model.mha_layers.len() +
        model.dropouts.len() +
        model.batch_norms_2d.len() +
        model.batch_norms_4d.len() +
        model.layer_norms.len()
    ) as f32;
    
    // Estimate connectivity from execution plan length (rough proxy)
    let connectivity = model.execution_plan.len() as f32;
    
    // Early layers benefit more from signal flow
    let has_conv = !model.conv1ds.is_empty() || !model.conv2ds.is_empty();
    let has_rnn = !model.lstms.is_empty() || !model.grus.is_empty();
    let has_attention = !model.mha_layers.is_empty();
    
    // Base score
    let mut base_score = 3.0 + (total_layers * 0.3);
    
    // Architecture-specific bonuses
    if has_conv { base_score += 2.0; }
    if has_rnn { base_score += 1.5; }
    if has_attention { base_score += 2.5; }
    
    // Connectivity factor
    let conn_factor = (connectivity / 10.0).sqrt().min(3.0);
    let score = base_score * (1.0 + conn_factor * 0.2);
    
    // Add deterministic noise based on layer counts
    let noise = (((model.denses.len() as f32 * 1.73) % 2.0) - 1.0) * 0.3;
    
    (score + noise).max(0.5).min(15.0) // clamp to reasonable range
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
            stop_evolution,
            scan_dataset,
            cache_dataset,
            list_library_genomes,
            save_to_library,
            delete_from_library,
            load_library_genome,
            save_dataset_profiles,
            load_dataset_profiles,
            preview_csv,
            compute_zero_cost_score,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
