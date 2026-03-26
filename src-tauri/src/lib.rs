#![recursion_limit = "512"]
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rfd::AsyncFileDialog;

use tokio;
type Backend = crate::backend::TrainBackend;
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use entities::{DynamicBatch, DynamicTensor, GraphModel, train_simple};
use serde::{Deserialize, Serialize};
use zero_cost_proxies::{ZeroCostConfig, ZeroCostMetrics};
use crate::dtos::TrainingProfiler;

pub mod data_loader;
pub mod backend;
pub mod dtos;
pub mod entities;
pub mod zero_cost_proxies;
pub mod csv_loader;
pub mod shape_inference;
pub mod orchestrator;
pub mod profiler;
pub mod pareto;
pub mod device_profiles;
pub mod device_library;
pub mod genealogy;
pub mod weight_io;
pub mod stopping_criteria;

/// Global session counter. Incremented by `stop_evolution`.
/// Each `evaluate_population` call captures a snapshot; if the current value
/// differs from the snapshot, the evaluation is cancelled.
static EVOLUTION_SESSION: AtomicU64 = AtomicU64::new(0);

/// Global cache: hash(genome_json + training_params) -> (loss, accuracy).
/// Persists across generations within the app lifetime. Any parameter difference = different hash.
static GENOME_EVAL_CACHE: std::sync::LazyLock<Mutex<HashMap<u64, (f32, f32)>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Global in-memory genealogy graph for lineage tracking across evolution operations.
static GENEALOGY_STORE: std::sync::LazyLock<Mutex<genealogy::GenealogyStore>> =
    std::sync::LazyLock::new(|| Mutex::new(genealogy::GenealogyStore::new()));

/// True while evaluate_population is running. Prevent overlapping runs that can
/// corrupt Burn/WGPU internal stream state.
static EVALUATION_ACTIVE: AtomicBool = AtomicBool::new(false);

// Canonical frontend image shape is HWC, while Burn internals operate on CHW/NCHW.
// This helper accepts either HWC or CHW and normalizes to CHW.
fn normalize_image_shape_to_internal_chw(shape: &[usize]) -> Vec<usize> {
    if shape.len() != 3 {
        return shape.to_vec();
    }

    let a = shape[0];
    let b = shape[1];
    let c = shape[2];

    let first_is_channel = a <= 4 && b > 4 && c > 4;
    let last_is_channel = c <= 4 && a > 4 && b > 4;

    if first_is_channel {
        vec![a, b, c]
    } else if last_is_channel {
        vec![c, a, b]
    } else {
        // Ambiguous case: default to HWC -> CHW for wire-level consistency.
        vec![c, a, b]
    }
}

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

#[tauri::command]
async fn register_founder(genome_id: String, generation: u32) -> Result<(), String> {
    let mut store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store
        .register_founder(genome_id, generation)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn register_mutation(
    parent_id: String,
    child_id: String,
    mutation_type: dtos::MutationType,
    generation: u32,
) -> Result<(), String> {
    let mut store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store
        .register_mutation(parent_id, child_id, mutation_type, generation)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn register_crossover(
    parent_a: String,
    parent_b: String,
    child_id: String,
    generation: u32,
) -> Result<(), String> {
    let mut store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store
        .register_crossover(parent_a, parent_b, child_id, generation)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_genealogy(genome_id: String) -> Result<genealogy::GenealogyPath, String> {
    let store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store.get_genealogy(&genome_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_ancestors(
    genome_id: String,
    depth: Option<u32>,
) -> Result<Vec<genealogy::GenomeLineageRecord>, String> {
    let store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store
        .get_ancestors(&genome_id, depth)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_descendants(
    genome_id: String,
    depth: Option<u32>,
) -> Result<Vec<genealogy::GenomeLineageRecord>, String> {
    let store = GENEALOGY_STORE
        .lock()
        .map_err(|e| format!("Genealogy store lock poisoned: {}", e))?;
    store
        .get_descendants(&genome_id, depth)
        .map_err(|e| e.to_string())
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
    type Backend = crate::backend::TrainBackend;
    let device = crate::backend::create_device();

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

    type Backend = crate::backend::TrainBackend;
    let device = crate::backend::create_device();

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
    // Helper to get batch size from first input tensor
    let get_batch_size = |batch: Option<&DynamicBatch<Backend>>| -> usize {
        batch
            .map(|b| match &b.inputs[0] {
                DynamicTensor::Dim2(t) => t.dims()[0],
                DynamicTensor::Dim3(t) => t.dims()[0],
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

#[derive(Clone, serde::Serialize)]
pub struct EvaluationResult {
    pub genome_id: String,
    pub loss: f32,
    pub accuracy: f32,
    pub profiler: Option<TrainingProfiler>,
}

async fn run_worker_job(
    app_handle: tauri::AppHandle,
    genome_index: usize,
    request: crate::dtos::WorkerTrainRequest,
) -> Result<EvaluationResult, String> {
    use tauri::Emitter;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    let current_exe = std::env::current_exe()
        .map_err(|e| format!("Failed to locate current executable for worker spawn: {}", e))?;

    let mut child = Command::new(current_exe)
        .arg("--train-worker")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn train worker process: {}", e))?;

    let request_json = serde_json::to_string(&request)
        .map_err(|e| format!("Failed to serialize worker request: {}", e))?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "Worker stdin is unavailable".to_string())?;
        stdin
            .write_all(request_json.as_bytes())
            .await
            .map_err(|e| format!("Failed writing worker request payload: {}", e))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| format!("Failed writing worker request newline: {}", e))?;
        stdin
            .flush()
            .await
            .map_err(|e| format!("Failed flushing worker stdin: {}", e))?;
    }

    if let Some(stderr) = child.stderr.take() {
        let job_id = request.job_id.clone();
        tokio::spawn(async move {
            let mut err_reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = err_reader.next_line().await {
                if !line.trim().is_empty() {
                    eprintln!("[worker-stderr idx={} job={}] {}", genome_index, job_id, line);
                }
            }
        });
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Worker stdout is unavailable".to_string())?;
    let mut reader = BufReader::new(stdout).lines();

    let mut final_result: Option<crate::dtos::WorkerTrainResult> = None;
    while let Some(line) = reader
        .next_line()
        .await
        .map_err(|e| format!("Failed reading worker stdout line: {}", e))?
    {
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(progress) = serde_json::from_str::<crate::dtos::WorkerTrainProgress>(&line) {
            let _ = app_handle.emit(
                "evaluating-batch-metrics",
                crate::entities::BatchMetrics {
                    genome_index,
                    epoch: progress.epoch,
                    batch: progress.batch,
                    total_batches: progress.total_batches,
                    step: progress.step,
                    total_steps: progress.total_steps,
                    elapsed_train_ms: progress.gpu_active_ms,
                    queue_wait_ms: progress.queue_wait_ms,
                    gpu_active_ms: progress.gpu_active_ms,
                    step_time_ms: progress.step_time_ms,
                    loss: 999.0,
                    accuracy: 0.0,
                },
            );
            continue;
        }

        if let Ok(result) = serde_json::from_str::<crate::dtos::WorkerTrainResult>(&line) {
            final_result = Some(result);
            break;
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Failed waiting for worker process: {}", e))?;
    if !status.success() {
        return Err(format!("Worker exited with non-success status: {}", status));
    }

    let result = final_result.ok_or_else(|| "Worker produced no result payload".to_string())?;
    if let Some(err) = result.error {
        return Err(format!("Worker job failed: {}", err));
    }

    Ok(EvaluationResult {
        genome_id: result.genome_id,
        loss: result.loss,
        accuracy: result.accuracy,
        profiler: result.profiler,
    })
}

#[tauri::command]
async fn evaluate_population(
    app_handle: tauri::AppHandle,
    genomes: Vec<String>,
    dataset_profile: String,
    batch_size: usize,
    per_genome_epochs: Vec<usize>,
    dataset_percent: usize,
    train_split: usize,
    val_split: usize,
    test_split: usize,
    genome_ids: Option<Vec<String>>,
    source_generation: Option<u32>,
    profiling: Option<crate::profiler::ProfilingConfig>,
    max_parallel_jobs: Option<usize>,
    execution_mode: Option<String>,
    memory_safety_margin_mb: Option<u64>,
) -> Result<Vec<EvaluationResult>, String> {
    if EVALUATION_ACTIVE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Another evaluation is already running. Wait for it to stop before starting a new run.".to_string());
    }

    struct EvaluationGuard;
    impl Drop for EvaluationGuard {
        fn drop(&mut self) {
            EVALUATION_ACTIVE.store(false, Ordering::SeqCst);
        }
    }
    let _evaluation_guard = EvaluationGuard;

    eprintln!(
        ">>> Entered evaluate_population. Preparing to process dataset profile '{}'...",
        dataset_profile
    );

    let requested_parallel_jobs = max_parallel_jobs.unwrap_or(1).max(1);
    let requested_execution_mode = execution_mode.unwrap_or_else(|| "sequential".to_string());
    let configured_safety_margin_mb = memory_safety_margin_mb.unwrap_or(128);

    eprintln!(
        ">>> Evaluation config: backend='{}', mode='{}', requested_parallel_jobs={}, safety_margin_mb={}",
        crate::backend::backend_name(),
        requested_execution_mode,
        requested_parallel_jobs,
        configured_safety_margin_mb
    );

    let parallel_requested = requested_parallel_jobs > 1 && requested_execution_mode != "sequential";
    if parallel_requested {
        eprintln!(
            ">>> Parallel evaluation requested: mode='{}', max_parallel_jobs={}, safety_margin_mb={}",
            requested_execution_mode,
            requested_parallel_jobs,
            configured_safety_margin_mb
        );
    }

    // 1. Capture the current session counter IMMEDIATELY.
    // Each evaluate_population call captures a snapshot; if the current value
    // differs from the snapshot (via stop_evolution), we abort.
    let session_snapshot = EVOLUTION_SESSION.load(Ordering::SeqCst);

    type Backend = crate::backend::TrainBackend;
    let device = crate::backend::create_device();
    println!(">>> Wgpu device initialized (default)");
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

    println!(">>> Starting DataLoader creation...");
    let loader = match crate::data_loader::DataLoader::new(profile.clone(), app_data_dir) {
        Ok(l) => {
            println!(">>> DataLoader created successfully!");
            l
        },
        Err(e) => {
            println!(">>> DataLoader creation FAILED: {}", e);
            return Err(e);
        },
    };

    println!(
        "Evaluating population of {} genomes on dataset: {} ({})\nBatch Size: {}, Epochs: {}",
        genomes.len(),
        profile.name,
        source_path_str,
        batch_size,
        per_genome_epochs.iter().max().unwrap_or(&0)
    );

    let mut valid_ids = loader.valid_sample_ids.clone();
    if valid_ids.is_empty() {
        return Err(format!(
            "No valid samples found matching all streams for dataset: {}",
            source_path_str
        ));
    }

    {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        valid_ids.shuffle(&mut rng);
    }

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

        if loader.stream_files.contains_key(stream_id) {
            for id in &valid_ids {
                let label = loader.get_class_label(stream_id, id).unwrap_or_else(|| "unknown".to_string());
                groups.entry(label).or_default().push(id.clone());
            }
        }

        for (label, mut members) in groups {
            {
                use rand::seq::SliceRandom;
                members.shuffle(&mut rand::rng());
            }
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
        {
            use rand::seq::SliceRandom;
            let mut local_rng = rand::rng();
            train_ids.shuffle(&mut local_rng);
            val_ids.shuffle(&mut local_rng);
            test_ids.shuffle(&mut local_rng);
        }
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
                let mut h = 64;
                let mut w = 64;
                let mut channels = 3;
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

                let external_shape = if stream.tensor_shape.len() == 3 {
                    stream.tensor_shape.clone()
                } else {
                    vec![h, w, channels]
                };

                input_overrides.push(normalize_image_shape_to_internal_chw(&external_shape));
            }
            crate::dtos::DataType::Vector => {
                let dim = stream.tensor_shape.get(0).cloned().unwrap_or(1);
                input_overrides.push(vec![dim]);
            }
            crate::dtos::DataType::TemporalSequence => {
                input_overrides.push(stream.tensor_shape.clone());
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
                    if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                        return Err("Evolution cancelled during batch assembly".to_string());
                    }

                    let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        $loader.load_sample(id, $dev)
                    }));

                    match load_result {
                        Ok(Ok(sample)) => {
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
                        Ok(Err(e)) => {
                            eprintln!("[ERROR] Sample '{}' dropped during load: {}", id, e);
                        }
                        Err(_) => {
                            eprintln!("[ERROR] Sample '{}' panicked during load; dropping sample", id);
                            return Err(format!(
                                "Sample '{}' panicked during load (Burn/WGPU state became invalid). Aborting evaluation.",
                                id
                            ));
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
                    let assembled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let inputs: Vec<crate::entities::DynamicTensor<Backend>> = batch_inputs
                            .into_iter()
                            .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                            .collect();
                        let targets: Vec<crate::entities::DynamicTensor<Backend>> = batch_targets
                            .into_iter()
                            .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                            .collect();
                        (inputs, targets)
                    }));

                    match assembled {
                        Ok((inputs, targets)) => {
                            assembled_batches.push(crate::entities::DynamicBatch {
                                inputs,
                                targets,
                            });
                        }
                        Err(_) => {
                            return Err("Batch concatenation panicked (Burn/WGPU state invalid). Aborting evaluation.".to_string());
                        }
                    }
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

    let autosave_hidden_archive = |genome_id: &str,
                                   genome_str: &str,
                                   loss: f32,
                                   accuracy: f32,
                                   profiler: Option<TrainingProfiler>| {
        let mut generation = source_generation.unwrap_or(0);
        let mut parent_genomes = Vec::new();

        if let Ok(store) = GENEALOGY_STORE.lock() {
            if let Some(record) = store.graph().nodes.get(genome_id) {
                generation = record.generation;
                parent_genomes = record.parent_ids.clone();
            }
        }

        let fitness_metrics = GenomeFitnessMetrics {
            loss,
            accuracy,
            adjusted_fitness: None,
            inference_latency_ms: profiler.as_ref().map(|p| p.inference_msec_per_sample),
            model_size_mb: None,
            training_time_ms: profiler.as_ref().map(|p| p.total_train_duration_ms),
        };

        let autosave_tags = vec!["hidden".to_string(), "autosave".to_string()];
        let autosave_name = format!("Hidden {}", genome_id);

        if let Err(e) = save_hidden_genome(
            genome_str,
            autosave_name,
            autosave_tags,
            generation,
            parent_genomes,
            fitness_metrics,
            profiler,
        ) {
            eprintln!(
                "[hidden_library] autosave failed for genome '{}': {}",
                genome_id, e
            );
        }
    };

    // 5. Evaluation Loop over each Genome
    const MAX_RETRIES: usize = 3;
    const RANDOM_CHANCE_THRESHOLD: f32 = 55.0;
    let evaluation_started_at = std::time::Instant::now();
    let selected_memory_mode = profiling
        .as_ref()
        .and_then(|cfg| cfg.memory_mode)
        .unwrap_or(crate::profiler::MemoryMode::Hybrid);

    let total_genomes = genomes.len();
    let evaluate_one_genome = |i: usize, genome_str: String, genome_id: String| {
        let app_handle = app_handle.clone();
        let device = device.clone();
        let dataset_profile = dataset_profile.clone();
        let input_overrides = input_overrides.clone();
        let output_overrides = output_overrides.clone();
        let train_batches = train_batches.clone();
        let val_batches = val_batches.clone();
        let test_batches = test_batches.clone();
        let per_genome_epochs = per_genome_epochs.clone();
        let requested_execution_mode = requested_execution_mode.clone();
        let requested_parallel_jobs = requested_parallel_jobs;
        let queued_at_ms = evaluation_started_at.elapsed().as_millis() as u64;

        async move {
            let queue_wait_ms = evaluation_started_at
                .elapsed()
                .as_millis()
                .saturating_sub(queued_at_ms as u128) as u64;

            // Check cancellation between genomes
            if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                return Ok((
                    i,
                    EvaluationResult {
                        genome_id,
                        loss: 999.0,
                        accuracy: 0.0,
                        profiler: None,
                    },
                ));
            }

            // Emit current genome index to the frontend for UI synchronization
            app_handle.emit("evaluating-genome", i).unwrap_or_else(|e| {
                eprintln!("Failed to emit evaluating-genome event: {}", e);
            });

            eprintln!(
                "\n===========================================================\nEvaluating Genome {}/{} (ID: genome_{})\n===========================================================",
                i + 1,
                total_genomes,
                i
            );
            eprintln!(
                ">>> Genome {} queue wait: {} ms (mode='{}', requested_workers={})",
                i,
                queue_wait_ms,
                requested_execution_mode,
                requested_parallel_jobs
            );

            // Compute a cache key from genome content + all training params
            let cache_key = {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                let epochs = *per_genome_epochs.get(i).unwrap_or(&0) as usize;
                genome_str.hash(&mut hasher);
                dataset_profile.hash(&mut hasher);
                batch_size.hash(&mut hasher);
                epochs.hash(&mut hasher);
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

                autosave_hidden_archive(&genome_id, &genome_str, cached_loss, cached_acc, None);

                return Ok((
                    i,
                    EvaluationResult {
                        genome_id,
                        loss: cached_loss,
                        accuracy: cached_acc,
                        profiler: None,
                    },
                ));
            }

            eprintln!("Genome JSON:\n{}", genome_str);

            match std::panic::catch_unwind(|| {
                crate::entities::GraphModel::<Backend>::build(
                    &genome_str,
                    &device,
                    Some(&input_overrides),
                    Some(&output_overrides),
                )
            }) {
                Ok(_initial_model) => {
                    let mut best_loss = 999.0_f32;
                    let mut best_acc = 0.0_f32;
                    let mut best_profiler: Option<TrainingProfiler> = None;
                    let mut best_model: Option<GraphModel<Backend>> = None;

                    for attempt in 0..MAX_RETRIES {
                        if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                            break;
                        }

                        let model = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            crate::entities::GraphModel::<Backend>::build(
                                &genome_str,
                                &device,
                                Some(&input_overrides),
                                Some(&output_overrides),
                            )
                        })) {
                            Ok(m) => m,
                            Err(_) => {
                                eprintln!(
                                    ">>> Genome {} attempt {} panicked while rebuilding model. Skipping attempt.",
                                    i,
                                    attempt + 1
                                );
                                continue;
                            }
                        };

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

                        let epochs = *per_genome_epochs.get(i).unwrap_or(&0) as usize;
                        let app_handle_clone = app_handle.clone();
                        let train_batches_local = train_batches.clone();
                        let val_batches_local = val_batches.clone();
                        let test_batches_local = test_batches.clone();

                        let training_task = tokio::task::spawn_blocking(move || {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let mut model_local = model;
                                let mut profiler = crate::profiler::ProfilerCollector::new();
                                profiler.set_memory_mode(selected_memory_mode);
                                profiler.set_queue_wait_ms(queue_wait_ms);

                                if epochs > 0 {
                                    crate::entities::run_eval_pass(
                                        &app_handle_clone,
                                        i,
                                        &mut model_local,
                                        &train_batches_local,
                                        epochs,
                                        0.001,
                                        is_classification,
                                        &EVOLUTION_SESSION,
                                        session_snapshot,
                                        queue_wait_ms,
                                        Some(&mut profiler),
                                    );
                                } else {
                                    println!(">>> Genome {}: Skipping training (0 epochs requested)", i);
                                }

                                if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                                    println!(
                                        ">>> Genome {} cancelled right after training. Skipping validation/test inside worker.",
                                        i
                                    );
                                    return (999.0, 0.0, profiler.finalize(), model_local);
                                }

                                let (val_loss, val_acc) = if !val_batches_local.is_empty() {
                                    crate::entities::run_validation_pass(
                                        &model_local,
                                        &val_batches_local,
                                        "Validation",
                                        is_classification,
                                        Some(&mut profiler),
                                    )
                                } else {
                                    (0.0, 0.0)
                                };

                                if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                                    println!(
                                        ">>> Genome {} cancelled after validation. Skipping test pass inside worker.",
                                        i
                                    );
                                    return (999.0, 0.0, profiler.finalize(), model_local);
                                }

                                let (loss, acc) = if !test_batches_local.is_empty() {
                                    crate::entities::run_validation_pass(
                                        &model_local,
                                        &test_batches_local,
                                        "Test",
                                        is_classification,
                                        Some(&mut profiler),
                                    )
                                } else if !val_batches_local.is_empty() {
                                    (val_loss, val_acc)
                                } else {
                                    crate::entities::run_validation_pass(
                                        &model_local,
                                        &train_batches_local,
                                        "Train",
                                        is_classification,
                                        None,
                                    )
                                };

                                (loss, acc, profiler.finalize(), model_local)
                            }))
                        })
                        .await;

                        let (final_loss, final_acc, profiler_result, trained_model) = match training_task {
                            Ok(Ok(tuple)) => tuple,
                            Ok(Err(_)) => {
                                return Err(format!(
                                    "Genome {} attempt {} panicked during training/validation. Aborting evaluate_population to avoid corrupted WGPU state.",
                                    i,
                                    attempt + 1
                                ));
                            }
                            Err(e) => {
                                eprintln!(
                                    ">>> Genome {} attempt {} failed to join training task: {}",
                                    i,
                                    attempt + 1,
                                    e
                                );
                                continue;
                            }
                        };

                        if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                            eprintln!(
                                ">>> Cancelled after training for Genome {}. Skipping val/test.",
                                i
                            );
                            break;
                        }

                        let attempt_is_better = attempt == 0 || final_acc > best_acc;
                        if attempt_is_better {
                            best_loss = final_loss;
                            best_acc = final_acc;
                            best_profiler = Some(profiler_result);
                            best_model = Some(trained_model);
                        }

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

                    if let Some(model) = best_model.as_ref() {
                        let cache_dir = get_weight_cache_dir();
                        if let Err(e) = fs::create_dir_all(&cache_dir).map_err(|err| err.to_string()) {
                            eprintln!("[weight_io] failed to ensure cache dir: {}", e);
                        } else if let Err(e) = crate::weight_io::save_weights(&genome_id, Some(model), &cache_dir) {
                            eprintln!(
                                "[weight_io] failed to checkpoint weights for genome '{}': {}",
                                genome_id, e
                            );
                        }
                    }

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

                    autosave_hidden_archive(
                        &genome_id,
                        &genome_str,
                        best_loss,
                        best_acc,
                        best_profiler.clone(),
                    );

                    if best_acc > RANDOM_CHANCE_THRESHOLD {
                        GENOME_EVAL_CACHE
                            .lock()
                            .unwrap()
                            .insert(cache_key, (best_loss, best_acc));
                    }

                    Ok((
                        i,
                        EvaluationResult {
                            genome_id,
                            loss: best_loss,
                            accuracy: best_acc,
                            profiler: best_profiler,
                        },
                    ))
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

                    Ok((
                        i,
                        EvaluationResult {
                            genome_id,
                            loss: 999.0,
                            accuracy: 0.0,
                            profiler: None,
                        },
                    ))
                }
            }
        }
    };

    if parallel_requested {
        let effective_parallel_jobs = requested_parallel_jobs.clamp(1, genomes.len().max(1));
        eprintln!(
            ">>> Running limited parallel evaluation with {} concurrent jobs (effective_gpu_workers={})",
            effective_parallel_jobs,
            effective_parallel_jobs
        );

        let mut indexed_results: Vec<Option<EvaluationResult>> = vec![None; genomes.len()];

        let process_worker_mode = requested_execution_mode == "parallel-safe-limited";
        let mut fallback_to_inprocess = false;

        if process_worker_mode && effective_parallel_jobs > 1 {
            eprintln!(
                ">>> Using process worker-pool mode (K={})",
                effective_parallel_jobs
            );
            eprintln!(
                ">>> Worker progress note: per-batch progress events are emitted only when workers output WorkerTrainProgress JSON lines."
            );

            let run_id = format!(
                "run-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0)
            );

            let mut join_set: tokio::task::JoinSet<(usize, Result<EvaluationResult, String>)> =
                tokio::task::JoinSet::new();
            let mut next_index = 0usize;
            let mut in_flight = 0usize;
            let mut worker_failures = 0usize;

            let spawn_next = |idx: usize,
                              join_set: &mut tokio::task::JoinSet<(usize, Result<EvaluationResult, String>)>| {
                let app = app_handle.clone();
                let genome_id = genome_ids
                    .as_ref()
                    .and_then(|ids| ids.get(idx))
                    .cloned()
                    .unwrap_or_else(|| format!("genome_{}", idx));
                let request = crate::dtos::WorkerTrainRequest {
                    job_id: format!("job-{}", idx),
                    run_id: run_id.clone(),
                    genome_id: genome_id.clone(),
                    genome_json: genomes[idx].clone(),
                    dataset_profile: dataset_profile.clone(),
                    batch_size,
                    epochs: *per_genome_epochs.get(idx).unwrap_or(&0),
                    dataset_percent,
                    train_split,
                    val_split,
                    test_split,
                    queue_entered_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };

                join_set.spawn(async move { (idx, run_worker_job(app, idx, request).await) });
            };

            while next_index < genomes.len() && in_flight < effective_parallel_jobs {
                spawn_next(next_index, &mut join_set);
                next_index += 1;
                in_flight += 1;
            }

            while in_flight > 0 {
                if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                    eprintln!(">>> Cancellation requested. Aborting worker pool and draining queue.");
                    join_set.abort_all();
                    fallback_to_inprocess = true;
                    break;
                }

                let joined = match join_set.join_next().await {
                    Some(value) => value,
                    None => break,
                };
                in_flight = in_flight.saturating_sub(1);

                match joined {
                    Ok((idx, Ok(result))) => {
                        #[derive(serde::Serialize, Clone)]
                        struct GenomeResult {
                            index: usize,
                            loss: f32,
                            accuracy: f32,
                        }
                        let _ = app_handle.emit(
                            "evaluating-genome-result",
                            GenomeResult {
                                index: idx,
                                loss: result.loss,
                                accuracy: result.accuracy,
                            },
                        );
                        indexed_results[idx] = Some(result);
                    }
                    Ok((idx, Err(err))) => {
                        worker_failures = worker_failures.saturating_add(1);
                        eprintln!(
                            ">>> Worker failed for genome {} (failure {}): {}",
                            idx, worker_failures, err
                        );

                        let genome_id = genome_ids
                            .as_ref()
                            .and_then(|ids| ids.get(idx))
                            .cloned()
                            .unwrap_or_else(|| format!("genome_{}", idx));

                        let retry = evaluate_one_genome(idx, genomes[idx].clone(), genome_id).await;
                        match retry {
                            Ok((resolved_idx, result)) => {
                                indexed_results[resolved_idx] = Some(result);
                            }
                            Err(retry_err) => {
                                eprintln!(
                                    ">>> In-process retry failed for genome {}: {}",
                                    idx, retry_err
                                );
                                fallback_to_inprocess = true;
                            }
                        }

                        if worker_failures >= 2 {
                            eprintln!(
                                ">>> Multiple worker failures detected. Switching remaining queue to in-process evaluation."
                            );
                            fallback_to_inprocess = true;
                        }
                    }
                    Err(join_err) => {
                        worker_failures = worker_failures.saturating_add(1);
                        eprintln!(">>> Worker task join failed: {}", join_err);
                        fallback_to_inprocess = true;
                    }
                }

                if fallback_to_inprocess {
                    join_set.abort_all();
                    break;
                }

                while next_index < genomes.len() && in_flight < effective_parallel_jobs {
                    spawn_next(next_index, &mut join_set);
                    next_index += 1;
                    in_flight += 1;
                }
            }
        } else {
            fallback_to_inprocess = true;
        }

        if fallback_to_inprocess {
            let reduced_parallel_jobs = effective_parallel_jobs.saturating_sub(1).max(1);
            eprintln!(
                ">>> Fallback policy active: evaluating remaining genomes in-process with parallelism {}",
                reduced_parallel_jobs
            );

            let mut chunk_start = 0usize;
            while chunk_start < genomes.len() {
                if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                    println!(">>> Evolution cancelled. Aborting remaining genomes.");
                    break;
                }

                let chunk_end = (chunk_start + reduced_parallel_jobs).min(genomes.len());
                let mut futures = Vec::new();

                for i in chunk_start..chunk_end {
                    if indexed_results[i].is_some() {
                        continue;
                    }
                    let genome_id = genome_ids
                        .as_ref()
                        .and_then(|ids| ids.get(i))
                        .cloned()
                        .unwrap_or_else(|| format!("genome_{}", i));
                    futures.push(evaluate_one_genome(i, genomes[i].clone(), genome_id));
                }

                let outcomes = futures::future::join_all(futures).await;
                for outcome in outcomes {
                    let (idx, result) = outcome?;
                    indexed_results[idx] = Some(result);
                }

                chunk_start = chunk_end;
            }
        }

        for result in indexed_results.into_iter().flatten() {
            results.push(result);
        }
    } else {
        eprintln!(
            ">>> Running sequential evaluation (effective_gpu_workers=1)"
        );
        for i in 0..genomes.len() {
            if EVOLUTION_SESSION.load(Ordering::SeqCst) != session_snapshot {
                println!(">>> Evolution cancelled. Aborting remaining genomes.");
                break;
            }

            let genome_id = genome_ids
                .as_ref()
                .and_then(|ids| ids.get(i))
                .cloned()
                .unwrap_or_else(|| format!("genome_{}", i));
            let (_, result) = evaluate_one_genome(i, genomes[i].clone(), genome_id).await?;
            results.push(result);
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
    pub data_type: Option<String>,   // "Image" | "Vector" | "Categorical" | "Text" | "TemporalSequence"
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

    // Step 1a: Check if any stream is CsvDataset вЂ” if so, use it as anchor
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
                        let sample_id = i.to_string();
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
                let data_type = cfg.data_type.as_deref().unwrap_or("Vector");
                
                let mut class_counts: HashMap<String, usize> = HashMap::new();
                let mut csv_samples = 0;

                // Save feature column count for logging
                let feature_col_count = feature_columns.len();
                let sample_mode = cfg.sample_mode.as_deref().unwrap_or("row");
                let effective_sample_mode = if stream_role == "Input" && data_type == "TemporalSequence" {
                    "temporal_window"
                } else {
                    sample_mode
                };
                // Never default temporal windows to 1; use 50 as safe fallback to match UI default.
                let window_size = if effective_sample_mode == "temporal_window" {
                    cfg.window_size.unwrap_or(50)
                } else {
                    cfg.window_size.unwrap_or(1)
                };
                
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
                        sample_mode: effective_sample_mode.to_string(),
                        feature_columns,
                        target_column: String::new(), // Empty for input streams
                        window_size: if effective_sample_mode == "temporal_window" { Some(window_size) } else { None },
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
                        // Only Target streams should report classes.
                        if stream_role == "Target" {
                            for class in &loader.discovered_classes {
                                *class_counts.entry(class.clone()).or_insert(0) += 1;
                            }
                        }
                        
                        eprintln!(
                            ">>> CSV scan success: stream '{}' ({}) loaded {} samples, classes: {:?}",
                            cfg.alias,
                            if stream_role == "Target" {
                                format!("Target, classes={}", class_counts.len())
                            } else {
                                format!("Input ({}, data_type={}), shape=[{}, {}]", effective_sample_mode, data_type, window_size, feature_col_count)
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
                    if effective_sample_mode == "temporal_window" {
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
                } else if effective_sample_mode == "temporal_window" || data_type == "TemporalSequence" {
                    "TemporalSequence"
                } else {
                    "Vector"
                };

                // Truncate valid_ids to match csv_samples if this is a temporal stream
                // since temporal windows produce fewer samples than rows.
                valid_ids.retain(|id| {
                    if let Ok(idx) = id.parse::<usize>() {
                        idx < csv_samples
                    } else {
                        false // Must be numeric to match CSV index
                    }
                });
                
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
    profile_json: String,
) -> Result<crate::data_loader::CacheResult, String> {
    use tauri::Manager;
    let profile: crate::dtos::DatasetProfile = serde_json::from_str(&profile_json)
        .map_err(|e| format!("Failed to parse profile JSON: {}", e))?;

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
    let total_valid_samples: usize = 0;

    // Collect input shapes from profile streams
    // After "Scan & Validate", tensor_shape is populated by the scan operation
    let input_streams: Vec<_> = profile
        .streams
        .iter()
        .filter(|s| s.role == "Input")
        .collect();

    // Extract shapes from Input streams (populated by scan)
    for stream in &input_streams {
        if !stream.tensor_shape.is_empty() {
            input_shapes.insert(stream.id.clone(), stream.tensor_shape.clone());
        } else {
            // Shape was not populated, likely because scan hasn't run yet
            issues.push(dtos::ValidationIssue {
                severity: dtos::ValidationSeverity::Error,
                component: "InputShape".to_string(),
                message: format!(
                    "Input stream '{}' has no inferred shape. Run 'Scan & Validate' first.",
                    stream.alias
                ),
                suggested_fix: Some("Click 'Scan & Validate' to infer shapes from your data".to_string()),
            });
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

            if idx == 0 {
                if let Some(num_c) = stream.num_classes {
                    if let Ok(shape) = crate::shape_inference::ShapeInference::infer_output_shape(stream, num_c) {
                        output_shape = Some(shape);
                    } else {
                        output_shape = Some(vec![num_c]);
                    }
                } else {
                    output_shape = Some(vec![0]); // Placeholder
                }
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
    if let Ok(custom_dir) = std::env::var("NEURAL_EVO_GENOMES_DIR") {
        return PathBuf::from(custom_dir);
    }

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

fn get_weight_cache_dir() -> PathBuf {
    get_genomes_dir().join("weights_cache")
}

fn current_unix_ms() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct GenomeFitnessMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub adjusted_fitness: Option<f32>,
    pub inference_latency_ms: Option<f32>,
    pub model_size_mb: Option<f32>,
    pub training_time_ms: Option<u64>,
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
    #[serde(default)]
    pub is_hidden: bool,
    #[serde(default)]
    pub source_generation: u32,
    #[serde(default)]
    pub parent_genomes: Vec<String>,
    #[serde(default)]
    pub fitness_metrics: Option<GenomeFitnessMetrics>,
    #[serde(default)]
    pub profiler_data: Option<TrainingProfiler>,
    #[serde(default)]
    pub created_at_unix_ms: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct HiddenLibraryQuery {
    pub generation_min: Option<u32>,
    pub generation_max: Option<u32>,
    pub accuracy_min: Option<f32>,
    pub accuracy_max: Option<f32>,
    pub latency_min_ms: Option<f32>,
    pub latency_max_ms: Option<f32>,
    pub model_size_min_mb: Option<f32>,
    pub model_size_max_mb: Option<f32>,
    pub parent_genome_id: Option<String>,
    pub created_after_unix_ms: Option<u64>,
    pub created_before_unix_ms: Option<u64>,
    pub limit: Option<usize>,
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

fn save_library_entry(genome_str: &str, mut entry: GenomeLibraryEntry) -> Result<GenomeLibraryEntry, String> {
    let dir = get_genomes_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let file_path = dir.join(format!("{}.evog", entry.id));
    fs::write(&file_path, genome_str).map_err(|e| e.to_string())?;

    let mut meta = read_meta();
    meta.retain(|e| e.id != entry.id);
    if entry.created_at_unix_ms == 0 {
        entry.created_at_unix_ms = current_unix_ms();
    }
    meta.push(entry.clone());
    write_meta(&meta)?;

    Ok(entry)
}

fn save_hidden_genome(
    genome_str: &str,
    name: String,
    tags: Vec<String>,
    source_generation: u32,
    parent_genomes: Vec<String>,
    fitness_metrics: GenomeFitnessMetrics,
    profiler_data: Option<TrainingProfiler>,
) -> Result<GenomeLibraryEntry, String> {
    let id = uuid::Uuid::new_v4().to_string();
    let (input_dims, output_dims, total_nodes, layer_types) = extract_genome_metadata(genome_str);
    let now = chrono::Utc::now();

    let entry = GenomeLibraryEntry {
        id,
        name,
        tags,
        created_at: now.to_rfc3339(),
        input_dims,
        output_dims,
        total_nodes,
        layer_types,
        best_loss: Some(fitness_metrics.loss),
        best_accuracy: Some(fitness_metrics.accuracy),
        is_hidden: true,
        source_generation,
        parent_genomes,
        fitness_metrics: Some(fitness_metrics),
        profiler_data,
        created_at_unix_ms: now.timestamp_millis().max(0) as u64,
    };

    save_library_entry(genome_str, entry)
}

fn matches_hidden_query(entry: &GenomeLibraryEntry, query: &HiddenLibraryQuery) -> bool {
    if !entry.is_hidden {
        return false;
    }

    if let Some(min_gen) = query.generation_min {
        if entry.source_generation < min_gen {
            return false;
        }
    }
    if let Some(max_gen) = query.generation_max {
        if entry.source_generation > max_gen {
            return false;
        }
    }

    let metrics = entry.fitness_metrics.as_ref();
    if let Some(min_acc) = query.accuracy_min {
        let acc = metrics.map(|m| m.accuracy).unwrap_or(0.0);
        if acc < min_acc {
            return false;
        }
    }
    if let Some(max_acc) = query.accuracy_max {
        let acc = metrics.map(|m| m.accuracy).unwrap_or(0.0);
        if acc > max_acc {
            return false;
        }
    }
    if let Some(min_latency) = query.latency_min_ms {
        let latency = metrics.and_then(|m| m.inference_latency_ms).unwrap_or(f32::MAX);
        if latency < min_latency {
            return false;
        }
    }
    if let Some(max_latency) = query.latency_max_ms {
        let latency = metrics.and_then(|m| m.inference_latency_ms).unwrap_or(f32::MAX);
        if latency > max_latency {
            return false;
        }
    }
    if let Some(min_size) = query.model_size_min_mb {
        let size = metrics.and_then(|m| m.model_size_mb).unwrap_or(f32::MAX);
        if size < min_size {
            return false;
        }
    }
    if let Some(max_size) = query.model_size_max_mb {
        let size = metrics.and_then(|m| m.model_size_mb).unwrap_or(f32::MAX);
        if size > max_size {
            return false;
        }
    }

    if let Some(parent) = query.parent_genome_id.as_ref() {
        if !entry.parent_genomes.iter().any(|p| p == parent) {
            return false;
        }
    }

    if let Some(after) = query.created_after_unix_ms {
        if entry.created_at_unix_ms < after {
            return false;
        }
    }
    if let Some(before) = query.created_before_unix_ms {
        if entry.created_at_unix_ms > before {
            return false;
        }
    }

    true
}

#[tauri::command]
async fn list_library_genomes() -> Result<Vec<GenomeLibraryEntry>, String> {
    let mut entries: Vec<GenomeLibraryEntry> = read_meta().into_iter().filter(|e| !e.is_hidden).collect();
    entries.sort_by_key(|e| e.created_at_unix_ms);
    Ok(entries)
}

#[tauri::command]
async fn list_hidden_library(query: Option<HiddenLibraryQuery>) -> Result<Vec<GenomeLibraryEntry>, String> {
    let q = query.unwrap_or_default();
    let mut entries: Vec<GenomeLibraryEntry> = read_meta()
        .into_iter()
        .filter(|e| matches_hidden_query(e, &q))
        .collect();

    entries.sort_by(|a, b| b.created_at_unix_ms.cmp(&a.created_at_unix_ms));
    if let Some(limit) = q.limit {
        entries.truncate(limit);
    }

    Ok(entries)
}

#[tauri::command]
async fn save_to_library(
    genome_str: String,
    name: String,
    tags: Vec<String>,
) -> Result<GenomeLibraryEntry, String> {
    let id = uuid::Uuid::new_v4().to_string();

    let (input_dims, output_dims, total_nodes, layer_types) = extract_genome_metadata(&genome_str);
    let now = chrono::Utc::now();

    let entry = GenomeLibraryEntry {
        id,
        name,
        tags,
        created_at: now.to_rfc3339(),
        input_dims,
        output_dims,
        total_nodes,
        layer_types,
        best_loss: None,
        best_accuracy: None,
        is_hidden: false,
        source_generation: 0,
        parent_genomes: vec![],
        fitness_metrics: None,
        profiler_data: None,
        created_at_unix_ms: now.timestamp_millis().max(0) as u64,
    };

    save_library_entry(&genome_str, entry)
}

#[tauri::command]
async fn unhide_genome(genome_id: String) -> Result<(), String> {
    let mut meta = read_meta();
    let mut found = false;

    for entry in &mut meta {
        if entry.id == genome_id {
            entry.is_hidden = false;
            found = true;
            break;
        }
    }

    if !found {
        return Err(format!("Hidden genome '{}' not found", genome_id));
    }

    write_meta(&meta)
}

#[tauri::command]
async fn delete_hidden_genome(genome_id: String) -> Result<(), String> {
    let dir = get_genomes_dir();
    let file_path = dir.join(format!("{}.evog", genome_id));
    if file_path.exists() {
        fs::remove_file(&file_path).map_err(|e| e.to_string())?;
    }

    let mut meta = read_meta();
    let old_len = meta.len();
    meta.retain(|e| e.id != genome_id || !e.is_hidden);

    if old_len == meta.len() {
        return Err(format!("Hidden genome '{}' not found", genome_id));
    }

    write_meta(&meta)
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

fn find_library_entry_for_genome(genome_id: &str) -> Option<GenomeLibraryEntry> {
    let mut entries = read_meta();
    if entries.is_empty() {
        return None;
    }

    // Prefer direct ID match first (library-selected export path).
    if let Some(entry) = entries.iter().find(|e| e.id == genome_id) {
        return Some(entry.clone());
    }

    // Evolution runtime typically tracks logical genome IDs while hidden archive uses UUID IDs.
    // We map by autosave naming convention and take the most recent match.
    let hidden_name = format!("Hidden {}", genome_id);
    entries
        .drain(..)
        .filter(|e| e.is_hidden && e.name == hidden_name)
        .max_by_key(|e| e.created_at_unix_ms)
}

#[derive(serde::Serialize)]
struct WeightExportResponse {
    weights_path: String,
    metadata_path: String,
    used_cached_weights: bool,
}

#[tauri::command]
async fn has_cached_weights(genome_id: String) -> Result<bool, String> {
    let cache_dir = get_weight_cache_dir();
    Ok(weight_io::load_weights(&genome_id, &cache_dir)?.is_some())
}

#[tauri::command]
async fn export_genome_with_weights(
    genome_id: String,
    output_path: String,
) -> Result<WeightExportResponse, String> {
    let output_dir = PathBuf::from(output_path);
    if output_dir.as_os_str().is_empty() {
        return Err("Output path is empty".to_string());
    }
    fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    let cache_dir = get_weight_cache_dir();
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;

    let cache_hit = weight_io::load_weights(&genome_id, &cache_dir)?;
    let used_cached_weights = cache_hit.is_some();

    let cache_weights_path = if let Some(path) = cache_hit {
        path
    } else {
        return Err(format!(
            "No cached trained weights found for genome '{}'. Run evaluation first to create a checkpoint.",
            genome_id
        ));
    };

    let export_weights_path = output_dir.join(format!("{}.mpk", genome_id));
    fs::copy(&cache_weights_path, &export_weights_path).map_err(|e| e.to_string())?;

    let entry = find_library_entry_for_genome(&genome_id);
    let fitness = entry.as_ref().and_then(|e| e.fitness_metrics.as_ref());
    let profiler = entry.as_ref().and_then(|e| e.profiler_data.as_ref());
    let lineage = if let Some(e) = entry.as_ref() {
        e.parent_genomes.clone()
    } else if let Ok(store) = GENEALOGY_STORE.lock() {
        store
            .graph()
            .nodes
            .get(&genome_id)
            .map(|n| n.parent_ids.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let objectives = weight_io::ExportObjectives {
        accuracy: fitness.map(|m| m.accuracy),
        inference_latency_ms: fitness.and_then(|m| m.inference_latency_ms),
        model_size_mb: fitness.and_then(|m| m.model_size_mb),
        train_duration_ms: fitness.and_then(|m| m.training_time_ms),
        device_profile_id: None,
        lineage,
    };

    let (_weights_path, metadata_path) =
        weight_io::export_with_metadata(&genome_id, &output_dir, &objectives, profiler)?;

    Ok(WeightExportResponse {
        weights_path: export_weights_path.to_string_lossy().to_string(),
        metadata_path: metadata_path.to_string_lossy().to_string(),
        used_cached_weights,
    })
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
    println!(">>> compute_zero_cost_score: start");
    let result = tokio::task::spawn_blocking(move || -> Result<ZeroCostMetrics, String> {
        let safe_run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<ZeroCostMetrics, String> {
                let device = crate::backend::create_device();
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

                // Build model from genome with explicit IO overrides inferred from serialized nodes.
                // This stabilizes shape semantics between CHW/HWC sources for zero-cost scoring.
                let mut input_overrides: Vec<Vec<usize>> = Vec::new();
                let mut output_overrides: Vec<Vec<usize>> = Vec::new();

                let mut parsing_connections = false;
                for line in genome_json.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
                    if line == "CONNECTIONS" {
                        parsing_connections = true;
                        continue;
                    }
                    if parsing_connections {
                        continue;
                    }

                    let parsed: crate::dtos::NodeDtoJSON = match serde_json::from_str(line) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    match parsed {
                        crate::dtos::NodeDtoJSON::Input { output_shape } => {
                            let shape: Vec<usize> = output_shape.iter().map(|&v| v as usize).collect();
                            if shape.len() == 3 {
                                input_overrides.push(normalize_image_shape_to_internal_chw(&shape));
                            } else {
                                input_overrides.push(shape);
                            }
                        }
                        crate::dtos::NodeDtoJSON::Output { input_shape } => {
                            output_overrides.push(input_shape.iter().map(|&v| v as usize).collect());
                        }
                        _ => {}
                    }
                }

                let input_overrides_ref = if input_overrides.is_empty() {
                    None
                } else {
                    Some(input_overrides.as_slice())
                };
                let output_overrides_ref = if output_overrides.is_empty() {
                    None
                } else {
                    Some(output_overrides.as_slice())
                };

                let model = GraphModel::<Backend>::build(
                    &genome_json,
                    &device,
                    input_overrides_ref,
                    output_overrides_ref,
                );

                // Create a minimal sample batch (1 sample with dummy data)
                let mut inputs = Vec::new();
                for shape in &model.input_shapes {
                    let tensor = if shape.len() == 1 {
                        DynamicTensor::Dim2(Tensor::<Backend, 2>::random(
                            [1, shape[0]],
                            Distribution::Normal(0.0, 1.0),
                            &device,
                        ))
                    } else if shape.len() == 2 {
                        DynamicTensor::Dim3(Tensor::<Backend, 3>::random(
                            [1, shape[0], shape[1]],
                            Distribution::Normal(0.0, 1.0),
                            &device,
                        ))
                    } else if shape.len() == 3 {
                        DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                            [1, shape[0], shape[1], shape[2]],
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
                        let class_idx = (rand::random::<f64>() * num_classes).floor();
                        DynamicTensor::Dim2(Tensor::<Backend, 2>::from_data([[class_idx]], &device))
                    } else if shape.len() == 2 {
                        DynamicTensor::Dim3(Tensor::<Backend, 3>::random(
                            [1, shape[0], shape[1]],
                            Distribution::Normal(0.0, 1.0),
                            &device,
                        ))
                    } else if shape.len() == 3 {
                        DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                            [1, shape[0], shape[1], shape[2]],
                            Distribution::Normal(0.0, 1.0),
                            &device,
                        ))
                    } else {
                        return Err("Unsupported output shape for zero-cost scoring".to_string());
                    };
                    targets.push(tensor);
                }

                let batch = DynamicBatch { inputs, targets };

                // Compute real SynFlow score
                let synflow_score = zero_cost_proxies::compute_synflow(&model, &batch);
                println!(">>> compute_zero_cost_score: score = {}", synflow_score);

                Ok(ZeroCostMetrics::from_synflow(synflow_score, &config))
            },
        ));

        match safe_run {
            Ok(res) => res,
            Err(_) => Err("Zero-cost scoring panicked (likely invalid shape or GPU memory pressure)".to_string()),
        }
    })
    .await
    .map_err(|e| format!("Task failed to join: {}", e))??;
    
    Ok(result)
}

#[tauri::command]
async fn compute_pareto_front(
    generation: u32,
    genomes: Vec<dtos::GenomeObjectives>,
    constraints: Option<device_profiles::DeviceResourceConstraints>,
    alpha: Option<f32>,
) -> Result<dtos::GenerationParetoFront, String> {
    let adjusted = if let Some(c) = constraints.as_ref() {
        let penalty_alpha = alpha.unwrap_or(1.0);
        genomes
            .iter()
            .cloned()
            .map(|mut g| {
                let (adjusted_accuracy, _) = device_profiles::score_fitness_with_device_constraints(
                    g.accuracy,
                    &g,
                    c,
                    penalty_alpha,
                );
                g.accuracy = adjusted_accuracy;
                g
            })
            .collect::<Vec<_>>()
    } else {
        genomes
    };

    Ok(pareto::compute_generation_pareto_front(generation, &adjusted))
}

#[tauri::command]
async fn get_device_profiles() -> Result<Vec<device_profiles::DeviceProfileDto>, String> {
    Ok(device_profiles::built_in_profiles())
}

#[tauri::command]
async fn validate_genome_for_device(
    genome_objectives: dtos::GenomeObjectives,
    constraints: device_profiles::DeviceResourceConstraints,
) -> Result<device_profiles::DeviceValidationResult, String> {
    Ok(device_profiles::validate_genome_for_device(
        &genome_objectives,
        &constraints,
    ))
}

#[tauri::command]
async fn apply_device_penalty(
    base_fitness: f32,
    violation_score: f32,
    alpha: f32,
) -> Result<f32, String> {
    Ok(device_profiles::apply_device_penalty(
        base_fitness,
        violation_score,
        alpha,
    ))
}

#[tauri::command]
async fn list_device_templates() -> Result<Vec<dtos::DeviceTemplateDto>, String> {
    device_library::list_device_templates()
}

#[tauri::command]
async fn create_device_template(
    input: dtos::CreateDeviceTemplateInput,
) -> Result<dtos::DeviceTemplateDto, String> {
    device_library::create_device_template(input)
}

#[tauri::command]
async fn update_device_template(
    id: String,
    patch: dtos::UpdateDeviceTemplatePatch,
) -> Result<dtos::DeviceTemplateDto, String> {
    device_library::update_device_template(id, patch)
}

#[tauri::command]
async fn delete_device_template(id: String) -> Result<(), String> {
    device_library::delete_device_template(id)
}

#[tauri::command]
async fn duplicate_device_template(
    id: String,
    new_name: String,
) -> Result<dtos::DeviceTemplateDto, String> {
    device_library::duplicate_device_template(id, new_name)
}

#[tauri::command]
async fn export_device_library(path: String) -> Result<usize, String> {
    device_library::export_device_library(path)
}

#[tauri::command]
async fn import_device_library(
    path: String,
    mode: dtos::DeviceLibraryImportMode,
) -> Result<Vec<dtos::DeviceTemplateDto>, String> {
    device_library::import_device_library(path, mode)
}

#[cfg(test)]
static TEST_ENV_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));


#[cfg(test)]
mod hidden_library_tests {
    use super::*;

    fn test_genome() -> String {
        [
            r#"{"node":"Input","params":{"output_shape":[28,28,1]}}"#,
            r#"{"node":"Dense","params":{"units":10,"activation":"softmax","use_bias":true}}"#,
            r#"{"node":"Output","params":{"input_shape":[10]}}"#,
            "CONNECTIONS",
            "0 1",
            "1 2",
        ]
        .join("\n")
    }

    fn with_test_storage() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hidden-library-test-{}", uuid::Uuid::new_v4()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        unsafe {
            std::env::set_var("NEURAL_EVO_GENOMES_DIR", dir.to_string_lossy().to_string());
        }
        dir
    }

    #[test]
    fn hidden_entry_serializes_and_deserializes() {
        let _guard = TEST_ENV_LOCK.lock().expect("test lock");
        let temp_dir = with_test_storage();

        let profiler = TrainingProfiler {
            train_start_ms: 1,
            first_batch_ms: 2,
            train_end_ms: 3,
            total_train_duration_ms: 4,
            val_start_ms: 5,
            val_end_ms: 6,
            val_duration_ms: 7,
            test_start_ms: 8,
            test_end_ms: 9,
            test_duration_ms: 10,
            peak_active_memory_mb: 11.0,
            peak_model_params_mb: 12.0,
            peak_gradient_mb: 13.0,
            peak_optim_state_mb: 14.0,
            peak_activation_mb: 15.0,
            samples_per_sec: 16.0,
            inference_msec_per_sample: 17.0,
            batch_count: 18,
            early_stop_epoch: Some(2),
            queue_wait_ms: 0,
            gpu_active_ms: 21,
            step_time_ms_ema: 1.2,
        };

        let saved = save_hidden_genome(
            &test_genome(),
            "entry".to_string(),
            vec!["hidden".to_string()],
            3,
            vec!["parent-a".to_string()],
            GenomeFitnessMetrics {
                loss: 0.2,
                accuracy: 92.0,
                adjusted_fitness: Some(90.0),
                inference_latency_ms: Some(1.7),
                model_size_mb: Some(2.5),
                training_time_ms: Some(333),
            },
            Some(profiler),
        )
        .expect("save hidden genome");

        let json = serde_json::to_string(&saved).expect("serialize");
        let parsed: GenomeLibraryEntry = serde_json::from_str(&json).expect("deserialize");

        assert!(parsed.is_hidden);
        assert_eq!(parsed.source_generation, 3);
        assert_eq!(parsed.parent_genomes, vec!["parent-a".to_string()]);
        assert!(parsed.profiler_data.is_some());
        assert!(parsed.fitness_metrics.is_some());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn save_list_unhide_delete_hidden_flow() {
        let _guard = TEST_ENV_LOCK.lock().expect("test lock");
        let temp_dir = with_test_storage();

        let saved = save_hidden_genome(
            &test_genome(),
            "entry".to_string(),
            vec!["hidden".to_string()],
            2,
            vec!["p1".to_string()],
            GenomeFitnessMetrics {
                loss: 0.3,
                accuracy: 88.0,
                adjusted_fitness: None,
                inference_latency_ms: Some(2.0),
                model_size_mb: Some(3.0),
                training_time_ms: Some(250),
            },
            None,
        )
        .expect("save hidden genome");

        let listed = futures::executor::block_on(list_hidden_library(None)).expect("list hidden");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, saved.id);

        futures::executor::block_on(unhide_genome(saved.id.clone())).expect("unhide");

        let listed_after_unhide = futures::executor::block_on(list_hidden_library(None)).expect("list hidden after unhide");
        assert_eq!(listed_after_unhide.len(), 0);

        let visible = futures::executor::block_on(list_library_genomes()).expect("visible list");
        assert_eq!(visible.len(), 1);

        // Re-hide for hidden delete path
        let mut meta = read_meta();
        for entry in &mut meta {
            if entry.id == saved.id {
                entry.is_hidden = true;
            }
        }
        write_meta(&meta).expect("write meta");

        futures::executor::block_on(delete_hidden_genome(saved.id.clone())).expect("delete hidden");

        let final_hidden = futures::executor::block_on(list_hidden_library(None)).expect("final list hidden");
        assert_eq!(final_hidden.len(), 0);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn hidden_filters_return_expected_subset() {
        let _guard = TEST_ENV_LOCK.lock().expect("test lock");
        let temp_dir = with_test_storage();

        let _a = save_hidden_genome(
            &test_genome(),
            "g1".to_string(),
            vec!["hidden".to_string()],
            1,
            vec!["root".to_string()],
            GenomeFitnessMetrics {
                loss: 0.4,
                accuracy: 70.0,
                adjusted_fitness: None,
                inference_latency_ms: Some(6.0),
                model_size_mb: Some(8.0),
                training_time_ms: Some(100),
            },
            None,
        )
        .expect("save g1");

        let _b = save_hidden_genome(
            &test_genome(),
            "g2".to_string(),
            vec!["hidden".to_string()],
            5,
            vec!["p2".to_string()],
            GenomeFitnessMetrics {
                loss: 0.1,
                accuracy: 95.0,
                adjusted_fitness: None,
                inference_latency_ms: Some(1.0),
                model_size_mb: Some(2.0),
                training_time_ms: Some(100),
            },
            None,
        )
        .expect("save g2");

        let query = HiddenLibraryQuery {
            generation_min: Some(3),
            generation_max: None,
            accuracy_min: Some(90.0),
            accuracy_max: None,
            latency_min_ms: None,
            latency_max_ms: Some(2.0),
            model_size_min_mb: None,
            model_size_max_mb: Some(3.0),
            parent_genome_id: Some("p2".to_string()),
            created_after_unix_ms: None,
            created_before_unix_ms: None,
            limit: Some(10),
        };

        let filtered = futures::executor::block_on(list_hidden_library(Some(query))).expect("filtered list");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "g2");

        let _ = fs::remove_dir_all(temp_dir);
    }
}

#[cfg(test)]
mod weight_export_command_tests {
    use super::*;

    fn with_test_storage() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("weight-export-test-{}", uuid::Uuid::new_v4()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        unsafe {
            std::env::set_var("NEURAL_EVO_GENOMES_DIR", dir.to_string_lossy().to_string());
        }
        dir
    }

    #[test]
    fn export_command_creates_mpk_and_metadata() {
        let _guard = TEST_ENV_LOCK.lock().expect("test lock");
        let temp_storage = with_test_storage();
        let output_dir = temp_storage.join("exports");

        let cache_dir = get_weight_cache_dir();
        fs::create_dir_all(&cache_dir).expect("create cache dir");
        let device = crate::backend::create_device();
        let genome = [
            r#"{"node":"Input","params":{"output_shape":[4]}}"#,
            r#"{"node":"Dense","params":{"units":3,"activation":"relu","use_bias":true}}"#,
            r#"{"node":"Output","params":{"input_shape":[3]}}"#,
            "CONNECTIONS",
            "0 1",
            "1 2",
        ]
        .join("\n");
        let model = GraphModel::<Backend>::build(&genome, &device, None, None);
        weight_io::save_weights("genome-export-1", Some(&model), &cache_dir)
            .expect("seed cached weights");

        let response = futures::executor::block_on(export_genome_with_weights(
            "genome-export-1".to_string(),
            output_dir.to_string_lossy().to_string(),
        ))
        .expect("export command succeeds");

        let weights_path = PathBuf::from(response.weights_path);
        let metadata_path = PathBuf::from(response.metadata_path);

        assert!(weights_path.exists());
        assert!(metadata_path.exists());

        let metadata_json = fs::read_to_string(metadata_path).expect("read metadata json");
        let metadata_value: serde_json::Value =
            serde_json::from_str(&metadata_json).expect("parse metadata json");
        assert_eq!(metadata_value["genome_id"], "genome-export-1");

        let _ = fs::remove_dir_all(temp_storage);
    }
}

/// Validate stopping criteria configuration
#[tauri::command]
async fn validate_stopping_criteria(
    criteria: Vec<dtos::StoppingCriterion>,
    policy: String,
) -> Result<(), String> {
    stopping_criteria::validate_stopping_config(&criteria, &policy)
}

/// Generate preview of stopping criteria behavior
#[tauri::command]
async fn generate_stopping_preview(
    criteria: Vec<dtos::StoppingCriterion>,
    policy: String,
) -> Result<stopping_criteria::StoppingPreview, String> {
    stopping_criteria::generate_stopping_preview(&criteria, &policy)
}

pub fn run_train_worker() {
    use std::io::{self, BufRead, Write};

    eprintln!(
        ">>> train-worker started (backend='{}')",
        crate::backend::backend_name()
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    type Backend = crate::backend::TrainBackend;
    let device = crate::backend::create_device();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(value) => value,
            Err(err) => {
                eprintln!(">>> worker stdin read failed: {}", err);
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let started = std::time::Instant::now();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let request: crate::dtos::WorkerTrainRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(err) => {
                let response = crate::dtos::WorkerTrainResult {
                    job_id: "unknown".to_string(),
                    genome_id: "unknown".to_string(),
                    loss: 999.0,
                    accuracy: 0.0,
                    profiler: None,
                    queue_wait_ms: 0,
                    wall_clock_ms: 0,
                    error: Some(format!("invalid worker request: {}", err)),
                };
                if let Ok(payload) = serde_json::to_string(&response) {
                    let _ = writeln!(stdout, "{}", payload);
                    let _ = stdout.flush();
                }
                continue;
            }
        };

        // Execute training pipeline
        let result = execute_worker_training(&request, &device);

        let wall_clock_ms = started.elapsed().as_millis() as u64;
        
        let response = crate::dtos::WorkerTrainResult {
            job_id: request.job_id,
            genome_id: request.genome_id,
            loss: result.0,
            accuracy: result.1,
            profiler: result.2,
            queue_wait_ms: now_ms.saturating_sub(request.queue_entered_ms),
            wall_clock_ms,
            error: result.3,
        };

        match serde_json::to_string(&response) {
            Ok(payload) => {
                if writeln!(stdout, "{}", payload).is_err() {
                    break;
                }
                if stdout.flush().is_err() {
                    break;
                }
            }
            Err(err) => {
                eprintln!(">>> worker serialize failed: {}", err);
                break;
            }
        }
    }
}

fn execute_worker_training(
    request: &crate::dtos::WorkerTrainRequest,
    device: &crate::backend::TrainDevice,
) -> (f32, f32, Option<crate::dtos::TrainingProfiler>, Option<String>) {
    use std::collections::HashMap;
    use std::panic::AssertUnwindSafe;

    eprintln!("[worker {}] Starting training for genome {}", request.job_id, request.genome_id);

    type Backend = crate::backend::TrainBackend;

    // 1. Load dataset profile
    let profiles_json = match crate::data_loader::load_dataset_profiles_sync() {
        Ok(json) => json,
        Err(e) => {
            eprintln!("[worker {}] Failed to load profiles: {}", request.job_id, e);
            return (999.0, 0.0, None, Some(format!("Failed to load dataset profiles: {}", e)));
        }
    };

    let root: crate::dtos::DatasetProfilesRoot = match serde_json::from_str(&profiles_json) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("[worker {}] Failed to parse profiles JSON: {}", request.job_id, e);
            return (999.0, 0.0, None, Some(format!("Failed to parse dataset profiles: {}", e)));
        }
    };

    let profile = match root.state.profiles.into_iter().find(|p| p.id == request.dataset_profile) {
        Some(p) => p,
        None => {
            let msg = format!("Dataset profile '{}' not found", request.dataset_profile);
            eprintln!("[worker {}] {}", request.job_id, msg);
            return (999.0, 0.0, None, Some(msg));
        }
    };

    let _source_path_str = match profile.source_path.clone() {
        Some(p) => p,
        None => {
            let msg = format!("Profile '{}' has no sourcePath", profile.name);
            eprintln!("[worker {}] {}", request.job_id, msg);
            return (999.0, 0.0, None, Some(msg));
        }
    };

    // 2. Create DataLoader
    let loader = match crate::data_loader::DataLoader::new(profile.clone(), None) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[worker {}] DataLoader creation failed: {}", request.job_id, e);
            return (999.0, 0.0, None, Some(format!("DataLoader creation failed: {}", e)));
        }
    };

    // 3. Build dataset split
    let mut valid_ids = loader.valid_sample_ids.clone();
    if valid_ids.is_empty() {
        let msg = format!("No valid samples found");
        eprintln!("[worker {}] {}", request.job_id, msg);
        return (999.0, 0.0, None, Some(msg));
    }

    {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        valid_ids.shuffle(&mut rng);
    }

    // Apply dataset percent
    let pct = request.dataset_percent.clamp(1, 100);
    let use_count = (valid_ids.len() * pct) / 100;
    let use_count = use_count.max(1);
    valid_ids.truncate(use_count);

    let total_split = (request.train_split + request.val_split + request.test_split).max(1) as f32;
    let train_ratio = request.train_split as f32 / total_split;
    let val_ratio = request.val_split as f32 / total_split;

    let mut train_ids = Vec::new();
    let mut val_ids = Vec::new();
    let mut test_ids = Vec::new();

    // Stratification (same as main path)
    let strat_stream_idx = profile.streams.iter().position(|s| {
        s.role == "Target" && matches!(s.data_type, crate::dtos::DataType::Categorical)
    });

    if let Some(s_idx) = strat_stream_idx {
        let stream_id = &profile.streams[s_idx].id;
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        if loader.stream_files.contains_key(stream_id) {
            for id in &valid_ids {
                let label = loader.get_class_label(stream_id, id).unwrap_or_else(|| "unknown".to_string());
                groups.entry(label).or_default().push(id.clone());
            }
        }

        for (_, mut members) in groups {
            {
                use rand::seq::SliceRandom;
                members.shuffle(&mut rand::rng());
            }
            let n = members.len();
            let t_count = ((n as f32) * train_ratio).round() as usize;
            let v_count = ((n as f32) * val_ratio).round() as usize;

            let t_count = t_count.min(n);
            let v_count = v_count.min(n - t_count);

            train_ids.extend(members.iter().take(t_count).cloned());
            val_ids.extend(members.iter().skip(t_count).take(v_count).cloned());
            test_ids.extend(members.iter().skip(t_count + v_count).cloned());
        }

        {
            use rand::seq::SliceRandom;
            let mut local_rng = rand::rng();
            train_ids.shuffle(&mut local_rng);
            val_ids.shuffle(&mut local_rng);
            test_ids.shuffle(&mut local_rng);
        }
    } else {
        let train_count = ((valid_ids.len() as f32) * train_ratio).round() as usize;
        let val_count = ((valid_ids.len() as f32) * val_ratio).round() as usize;

        let train_count = train_count.min(valid_ids.len());
        let val_count = val_count.min(valid_ids.len() - train_count);

        train_ids = valid_ids.iter().take(train_count).cloned().collect();
        val_ids = valid_ids.iter().skip(train_count).take(val_count).cloned().collect();
        test_ids = valid_ids.iter().skip(train_count + val_count).cloned().collect();
    }

    eprintln!("[worker {}] Split: {} train, {} val, {} test", request.job_id, train_ids.len(), val_ids.len(), test_ids.len());

    // 4. Get input/output overrides
    let input_stream_indices: Vec<usize> = profile.streams.iter().enumerate()
        .filter(|(_, s)| s.role == "Input")
        .map(|(i, _)| i)
        .collect();
    let target_stream_indices: Vec<usize> = profile.streams.iter().enumerate()
        .filter(|(_, s)| s.role == "Target")
        .map(|(i, _)| i)
        .collect();

    if input_stream_indices.is_empty() || target_stream_indices.is_empty() {
        return (999.0, 0.0, None, Some("No Input/Target streams found".to_string()));
    }

    let mut input_overrides = Vec::new();
    for &idx in &input_stream_indices {
        let stream = &profile.streams[idx];
        match stream.data_type {
            crate::dtos::DataType::Image => {
                let mut h = 64; let mut w = 64; let mut channels = 3;
                if let Some(prep) = &stream.preprocessing {
                    if let Some(vision) = &prep.vision {
                        if vision.resize.len() == 2 {
                            w = vision.resize[0] as usize;
                            h = vision.resize[1] as usize;
                        }
                        if vision.grayscale { channels = 1; }
                    }
                }
                let external_shape = if stream.tensor_shape.len() == 3 {
                    stream.tensor_shape.clone()
                } else {
                    vec![h, w, channels]
                };
                input_overrides.push(normalize_image_shape_to_internal_chw(&external_shape));
            }
            crate::dtos::DataType::Vector => {
                let dim = stream.tensor_shape.get(0).cloned().unwrap_or(1);
                input_overrides.push(vec![dim]);
            }
            crate::dtos::DataType::TemporalSequence => {
                input_overrides.push(stream.tensor_shape.clone());
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
            let num_classes = loader.stream_classes.get(&idx).cloned().unwrap_or(1);
            output_overrides.push(vec![num_classes]);
        } else {
            let dim = stream.tensor_shape.get(0).cloned().unwrap_or(1);
            output_overrides.push(vec![dim]);
        }
    }

    // 5. Build batches
    let mut assemble_batches = |ids: &[String]| -> Result<Vec<crate::entities::DynamicBatch<Backend>>, String> {
        let mut assembled_batches: Vec<crate::entities::DynamicBatch<Backend>> = Vec::new();
        for chunk in ids.chunks(request.batch_size) {
            let mut batch_inputs: Vec<Vec<crate::entities::DynamicTensor<Backend>>> = 
                vec![Vec::new(); input_stream_indices.len()];
            let mut batch_targets: Vec<Vec<crate::entities::DynamicTensor<Backend>>> = 
                vec![Vec::new(); target_stream_indices.len()];

            for id in chunk {
                let load_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    loader.load_sample(id, device)
                }));

                match load_result {
                    Ok(Ok(sample)) => {
                        for (i, &stream_idx) in input_stream_indices.iter().enumerate() {
                            if let Some(t) = sample.stream_tensors.get(&stream_idx) {
                                batch_inputs[i].push(t.clone());
                            }
                        }
                        for (i, &stream_idx) in target_stream_indices.iter().enumerate() {
                            if let Some(t) = sample.stream_tensors.get(&stream_idx) {
                                batch_targets[i].push(t.clone());
                            }
                        }
                    }
                    Ok(Err(_)) => { /* skip sample */ }
                    Err(_) => {
                        return Err("Sample panicked during load".to_string());
                    }
                }
            }

            if batch_inputs.iter().all(|list| !list.is_empty()) && 
               batch_targets.iter().all(|list| !list.is_empty()) {
                use crate::entities::concat_dynamic_tensors;
                let assembled = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    let inputs: Vec<crate::entities::DynamicTensor<Backend>> = batch_inputs
                        .into_iter()
                        .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                        .collect();
                    let targets: Vec<crate::entities::DynamicTensor<Backend>> = batch_targets
                        .into_iter()
                        .map(|tensors| concat_dynamic_tensors::<Backend>(tensors))
                        .collect();
                    (inputs, targets)
                }));

                match assembled {
                    Ok((inputs, targets)) => {
                        assembled_batches.push(crate::entities::DynamicBatch { inputs, targets });
                    }
                    Err(_) => {
                        return Err("Batch concatenation panicked".to_string());
                    }
                }
            }
        }
        Ok(assembled_batches)
    };

    let train_batches = match assemble_batches(&train_ids) {
        Ok(b) => b,
        Err(e) => {
            return (999.0, 0.0, None, Some(format!("Batch assembly failed: {}", e)));
        }
    };

    if train_batches.is_empty() {
        return (999.0, 0.0, None, Some("No training batches assembled".to_string()));
    }

    let val_batches = match assemble_batches(&val_ids) {
        Ok(b) => b,
        Err(_) => Vec::new(),
    };

    let test_batches = match assemble_batches(&test_ids) {
        Ok(b) => b,
        Err(_) => Vec::new(),
    };

    // 6. Build model from genome JSON
    let model = match std::panic::catch_unwind(AssertUnwindSafe(|| {
        crate::entities::GraphModel::<Backend>::build(
            &request.genome_json,
            device,
            Some(&input_overrides),
            Some(&output_overrides),
        )
    })) {
        Ok(m) => m,
        Err(_) => {
            return (999.0, 0.0, None, Some("Genome compilation failed".to_string()));
        }
    };

    // 7. Run evaluation (simplified worker path - no training yet)
    let mut profiler = crate::profiler::ProfilerCollector::new();
    let queue_wait_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
        .saturating_sub(request.queue_entered_ms);
    profiler.set_queue_wait_ms(queue_wait_ms);

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let model_local = model;
        profiler.mark_train_start();
        profiler.mark_train_end();

        // Validation phase
        let (val_loss, val_acc) = if !val_batches.is_empty() {
            profiler.mark_val_start();
            let mut val_loss_sum = 0.0;
            let mut val_correct = 0;
            let mut val_total = 0;

            for batch in &val_batches {
                let cloned_inputs: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.inputs.iter().map(|t| t.clone()).collect();
                let cloned_targets: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.targets.iter().map(|t| t.clone()).collect();

                let predictions = model_local.forward_internal(&cloned_inputs, false, false);
                let loss = model_local.compute_loss(&predictions, &cloned_targets);
                let loss_val = loss.into_data().to_vec::<f32>().unwrap_or(vec![0.0])[0];
                val_loss_sum += loss_val;

                let (correct, count) = crate::entities::compute_accuracy(&predictions, &cloned_targets, is_classification);
                val_correct += correct;
                val_total += count;

                profiler.record_inference_samples(count);
            }
            profiler.mark_val_end();

            let avg_loss = if !val_batches.is_empty() { val_loss_sum / val_batches.len() as f32 } else { 0.0 };
            let avg_acc = if val_total > 0 { val_correct as f32 / val_total as f32 } else { 0.0 };
            (avg_loss, avg_acc)
        } else {
            (0.0, 0.0)
        };

        // Test phase
        let (loss, acc) = if !test_batches.is_empty() {
            profiler.mark_test_start();
            let mut test_loss_sum = 0.0;
            let mut test_correct = 0;
            let mut test_total = 0;

            for batch in &test_batches {
                let cloned_inputs: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.inputs.iter().map(|t| t.clone()).collect();
                let cloned_targets: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.targets.iter().map(|t| t.clone()).collect();

                let predictions = model_local.forward_internal(&cloned_inputs, false, false);
                let loss = model_local.compute_loss(&predictions, &cloned_targets);
                let loss_val = loss.into_data().to_vec::<f32>().unwrap_or(vec![0.0])[0];
                test_loss_sum += loss_val;

                let (correct, count) = crate::entities::compute_accuracy(&predictions, &cloned_targets, is_classification);
                test_correct += correct;
                test_total += count;

                profiler.record_inference_samples(count);
            }
            profiler.mark_test_end();

            let avg_loss = if !test_batches.is_empty() { test_loss_sum / test_batches.len() as f32 } else { 0.0 };
            let avg_acc = if test_total > 0 { test_correct as f32 / test_total as f32 } else { 0.0 };
            (avg_loss, avg_acc)
        } else if !val_batches.is_empty() {
            (val_loss, val_acc)
        } else {
            // Fallback to train loss if no val/test
            let mut train_loss_sum = 0.0;
            let mut train_correct = 0;
            let mut train_total = 0;

            for batch in &train_batches {
                let cloned_inputs: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.inputs.iter().map(|t| t.clone()).collect();
                let cloned_targets: Vec<crate::entities::DynamicTensor<Backend>> =
                    batch.targets.iter().map(|t| t.clone()).collect();

                let predictions = model_local.forward_internal(&cloned_inputs, false, false);
                let loss = model_local.compute_loss(&predictions, &cloned_targets);
                let loss_val = loss.into_data().to_vec::<f32>().unwrap_or(vec![0.0])[0];
                train_loss_sum += loss_val;

                let (correct, count) = crate::entities::compute_accuracy(&predictions, &cloned_targets, is_classification);
                train_correct += correct;
                train_total += count;
            }

            let avg_loss = if !train_batches.is_empty() { train_loss_sum / train_batches.len() as f32 } else { 0.0 };
            let avg_acc = if train_total > 0 { train_correct as f32 / train_total as f32 } else { 0.0 };
            (avg_loss, avg_acc)
        };

        (loss, acc, profiler.finalize())
    }));

    match result {
        Ok((loss, acc, profiler_result)) => {
            eprintln!("[worker {}] Evaluation complete: loss={:.4}, acc={:.2}%", request.job_id, loss, acc * 100.0);
            (loss, acc, Some(profiler_result), None)
        }
        Err(_) => {
            (999.0, 0.0, None, Some("Evaluation panicked".to_string()))
        }
    }
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
            validate_dataset_profile,
            list_library_genomes,
            list_hidden_library,
            save_to_library,
            unhide_genome,
            delete_hidden_genome,
            delete_from_library,
            load_library_genome,
            export_genome_with_weights,
            has_cached_weights,
            save_dataset_profiles,
            load_dataset_profiles,
            preview_csv,
            compute_zero_cost_score,
            compute_pareto_front,
            get_device_profiles,
            validate_genome_for_device,
            apply_device_penalty,
            list_device_templates,
            create_device_template,
            update_device_template,
            delete_device_template,
            duplicate_device_template,
            export_device_library,
            import_device_library,
            register_founder,
            register_mutation,
            register_crossover,
            get_genealogy,
            get_ancestors,
            get_descendants,
            validate_stopping_criteria,
            generate_stopping_preview,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
