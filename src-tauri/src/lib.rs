use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rfd::AsyncFileDialog;

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use entities::{DynamicBatch, DynamicTensor, GraphModel, train_simple};

/// Global cache for preprocessed pixel data that persists across evaluate_population calls.
/// Key: (dataset_profile_id, target_w, target_h, is_grayscale)
/// Value: vector of (pixels_f32, label_id) grouped into batch-sized chunks
static DATASET_CACHE: std::sync::LazyLock<Mutex<Option<DatasetCacheEntry>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

struct DatasetCacheEntry {
    key: (String, u32, u32, bool, usize, usize, usize, usize), // (profile_id, w, h, grayscale, dataset_pct, train_split, val_split, test_split)
    train_batches: Vec<Vec<(Vec<f32>, usize)>>, // chunks of (pixels, label) for training
    val_batches: Vec<Vec<(Vec<f32>, usize)>>,   // chunks of (pixels, label) for validation
    test_batches: Vec<Vec<(Vec<f32>, usize)>>,  // chunks of (pixels, label) for testing
}

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
        .ok_or_else(|| format!("Profile '{}' has no sourcePath defined", profile.name))?;
    let source_dir = std::path::Path::new(&source_path_str);

    if !source_dir.exists() {
        return Err(format!(
            "Source directory does not exist: {}",
            source_path_str
        ));
    }

    // Find the input stream to get resizing parameters
    let input_stream = profile
        .streams
        .iter()
        .find(|s| s.role == "Input" && s.alias.contains("Stream"))
        // Fallback: just find the first input stream
        .or_else(|| profile.streams.iter().find(|s| s.role == "Input"))
        .ok_or_else(|| format!("Profile '{}' has no Input stream defined", profile.name))?;

    let mut target_h = 64;
    let mut target_w = 64;
    let mut is_grayscale = false;

    if let Some(prep) = &input_stream.preprocessing {
        if let Some(vision) = &prep.vision {
            if vision.resize.len() == 2 {
                target_w = vision.resize[0];
                target_h = vision.resize[1];
            }
            is_grayscale = vision.grayscale;
        }
    }

    println!(
        "Evaluating population of {} genomes on dataset: {} ({})\nTarget Dims: {}x{}, Grayscale: {}, Batch Size: {}, Epochs: {}",
        genomes.len(),
        profile.name,
        source_path_str,
        target_w,
        target_h,
        is_grayscale,
        batch_size,
        eval_epochs
    );

    // 2. Discover dataset structure (FolderMapping logic)
    // Map class names (folder names) to integer labels
    let mut class_to_id: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut all_images: Vec<(std::path::PathBuf, usize)> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(source_dir) {
        let mut class_id = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(folder_name) = path.file_name().and_then(|n| n.to_str()) {
                    class_to_id.insert(folder_name.to_string(), class_id);
                    // Read images in this class folder
                    if let Ok(images) = std::fs::read_dir(&path) {
                        for img_entry in images.flatten() {
                            let img_path = img_entry.path();
                            if img_path.is_file() {
                                if let Some(ext) = img_path.extension().and_then(|e| e.to_str()) {
                                    let ext = ext.to_lowercase();
                                    if ext == "jpg"
                                        || ext == "jpeg"
                                        || ext == "png"
                                        || ext == "webp"
                                    {
                                        all_images.push((img_path, class_id));
                                    }
                                }
                            }
                        }
                    }
                    class_id += 1;
                }
            }
        }
    }

    if all_images.is_empty() {
        return Err(format!(
            "No images found in dataset directory: {}",
            source_path_str
        ));
    }

    println!(
        "Discovered {} total images across {} classes.",
        all_images.len(),
        class_to_id.len()
    );

    // 3. Optional: Shuffle and Stratified Truncation
    let pct = dataset_percent.clamp(1, 100);
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();

    // Group images by class
    let mut class_groups: std::collections::HashMap<usize, Vec<(std::path::PathBuf, usize)>> =
        std::collections::HashMap::new();
    for img in all_images {
        class_groups.entry(img.1).or_default().push(img);
    }

    // Shuffle within each class and take `pct` from each class
    let mut stratified_images = Vec::new();
    let mut total_original = 0;
    for (_, mut group) in class_groups {
        total_original += group.len();
        group.shuffle(&mut rng);
        let use_count = (group.len() * pct) / 100;
        let use_count = use_count.max(1); // ensure at least 1 if >0 originally
        stratified_images.extend(group.into_iter().take(use_count));
    }

    // Shuffle the final stratified list so classes are mixed
    stratified_images.shuffle(&mut rng);
    let all_images = stratified_images;

    println!(
        "Using {}% of dataset: {} images (out of {} total)",
        pct,
        all_images.len(),
        total_original
    );

    // 3.5 Prepare shape overrides from the dataset profile
    let channels = if is_grayscale { 1 } else { 3 };
    let input_override = vec![channels, target_h as usize, target_w as usize]; // [C, H, W]
    let output_override = vec![class_to_id.len()]; // number of classes
    eprintln!(
        ">>> Dataset shape overrides: Input=[{}, {}, {}], Output=[{}]",
        channels,
        target_h,
        target_w,
        class_to_id.len()
    );

    // 3.6 Prepare Batch Caching with global persistence
    let cache_key = (
        dataset_profile.clone(),
        target_w,
        target_h,
        is_grayscale,
        dataset_percent,
        train_split,
        val_split,
        test_split,
    );

    // Check if we already have cached pixel data from a previous generation
    let cached_pixel_batches: (
        Vec<Vec<(Vec<f32>, usize)>>,
        Vec<Vec<(Vec<f32>, usize)>>,
        Vec<Vec<(Vec<f32>, usize)>>,
    );
    {
        let cache_guard = DATASET_CACHE.lock().unwrap();
        if let Some(entry) = cache_guard.as_ref() {
            if entry.key == cache_key {
                eprintln!(
                    ">>> Using cached dataset from previous generation ({} train batches, {} val batches, {} test batches)",
                    entry.train_batches.len(),
                    entry.val_batches.len(),
                    entry.test_batches.len()
                );
                cached_pixel_batches = (
                    entry.train_batches.clone(),
                    entry.val_batches.clone(),
                    entry.test_batches.clone(),
                );
            } else {
                eprintln!(">>> Dataset parameters changed, cache invalidated. Will re-process.");
                cached_pixel_batches = (Vec::new(), Vec::new(), Vec::new());
            }
        } else {
            cached_pixel_batches = (Vec::new(), Vec::new(), Vec::new());
        }
    }

    // 4. Evaluation Loop over each Genome
    for (i, genome_str) in genomes.iter().enumerate() {
        eprintln!(
            "\n===========================================================\nEvaluating Genome {}/{} (ID: genome_{})\n===========================================================",
            i + 1,
            genomes.len(),
            i
        );

        let input_overrides = vec![input_override.clone()];
        let output_overrides = vec![output_override.clone()];
        match std::panic::catch_unwind(|| {
            GraphModel::<Backend>::build(
                genome_str,
                &device,
                Some(&input_overrides),
                Some(&output_overrides),
            )
        }) {
            Ok(mut model) => {
                use burn::tensor::Tensor;

                // If we have a global cache hit, use it directly
                // Otherwise, process images with Rayon and save to global cache
                let mut train_pixel_batches = Vec::new();
                let mut val_pixel_batches = Vec::new();
                let mut test_pixel_batches = Vec::new();

                if !cached_pixel_batches.0.is_empty() {
                    // Reuse cache from a previous generation
                    train_pixel_batches = cached_pixel_batches.0.clone();
                    val_pixel_batches = cached_pixel_batches.1.clone();
                    test_pixel_batches = cached_pixel_batches.2.clone();
                } else if i == 0 {
                    // Cache Miss - need to load and process images
                    eprintln!(
                        ">>> [Genome 0] Parallel processing {} images into pixel batches...",
                        all_images.len()
                    );

                    // 1. Perform stratified split into train, val, and test images FIRST
                    let mut train_images = Vec::new();
                    let mut val_images = Vec::new();
                    let mut test_images = Vec::new();

                    // Group again to split perfectly
                    let mut class_groups: std::collections::HashMap<
                        usize,
                        Vec<&(std::path::PathBuf, usize)>,
                    > = std::collections::HashMap::new();
                    for img in &all_images {
                        class_groups.entry(img.1).or_default().push(img);
                    }

                    let total_split = (train_split + val_split + test_split).max(1) as f32;
                    let train_ratio = train_split as f32 / total_split;
                    let val_ratio = val_split as f32 / total_split;

                    let mut rng = rand::rng();
                    for (_, mut group) in class_groups {
                        use rand::seq::SliceRandom;
                        group.shuffle(&mut rng);

                        let train_count = ((group.len() as f32) * train_ratio).round() as usize;
                        let val_count = ((group.len() as f32) * val_ratio).round() as usize;

                        let train_count = train_count.min(group.len());
                        let val_count = val_count.min(group.len() - train_count);
                        let test_count = group.len() - train_count - val_count;

                        train_images.extend(group.iter().take(train_count).cloned().cloned());
                        val_images.extend(
                            group
                                .iter()
                                .skip(train_count)
                                .take(val_count)
                                .cloned()
                                .cloned(),
                        );
                        test_images.extend(
                            group
                                .iter()
                                .skip(train_count + val_count)
                                .take(test_count)
                                .cloned()
                                .cloned(),
                        );
                    }

                    use rand::seq::SliceRandom;
                    train_images.shuffle(&mut rng);
                    val_images.shuffle(&mut rng);
                    test_images.shuffle(&mut rng);

                    eprintln!(
                        ">>> Stratified split: {} train images, {} val images, {} test images",
                        train_images.len(),
                        val_images.len(),
                        test_images.len()
                    );

                    use rayon::prelude::*;

                    // Helper closure to process a chunk
                    let process_chunk =
                        |chunk: &[(std::path::PathBuf, usize)]| -> Vec<Vec<(Vec<f32>, usize)>> {
                            let mut local_batches = Vec::new();
                            for inner_chunk in chunk.chunks(batch_size) {
                                let processed_items: Vec<_> = inner_chunk
                                    .par_iter()
                                    .filter_map(|(img_path, label_id)| {
                                        if let Ok(img) = image::open(img_path) {
                                            use image::imageops::FilterType;
                                            let img = img.resize_exact(
                                                target_w,
                                                target_h,
                                                FilterType::Triangle,
                                            );

                                            let mut pixels = Vec::with_capacity(
                                                (target_w * target_h) as usize * channels,
                                            );
                                            if is_grayscale {
                                                let img_gray = img.to_luma8();
                                                for y in 0..target_h {
                                                    for x in 0..target_w {
                                                        pixels.push(
                                                            img_gray.get_pixel(x, y)[0] as f32
                                                                / 255.0,
                                                        );
                                                    }
                                                }
                                            } else {
                                                let img_rgb = img.to_rgb8();
                                                for c_idx in 0..3 {
                                                    for y in 0..target_h {
                                                        for x in 0..target_w {
                                                            pixels.push(
                                                                img_rgb.get_pixel(x, y)[c_idx]
                                                                    as f32
                                                                    / 255.0,
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            Some((pixels, *label_id))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();

                                if !processed_items.is_empty() {
                                    local_batches.push(processed_items);
                                }
                            }
                            local_batches
                        };

                    train_pixel_batches = process_chunk(&train_images);
                    val_pixel_batches = process_chunk(&val_images);
                    test_pixel_batches = process_chunk(&test_images);

                    eprintln!(
                        ">>> Finished processing! {} train pixel batches, {} val pixel batches, {} test pixel batches ready.",
                        train_pixel_batches.len(),
                        val_pixel_batches.len(),
                        test_pixel_batches.len()
                    );

                    // Save to global cache for future generations
                    {
                        let mut cache_guard = DATASET_CACHE.lock().unwrap();
                        *cache_guard = Some(DatasetCacheEntry {
                            key: cache_key.clone(),
                            train_batches: train_pixel_batches.clone(),
                            val_batches: val_pixel_batches.clone(),
                            test_batches: test_pixel_batches.clone(),
                        });
                    }
                } else {
                    // Genome 1+ within the same generation: reuse what genome 0 cached
                    let cache_guard = DATASET_CACHE.lock().unwrap();
                    if let Some(entry) = cache_guard.as_ref() {
                        train_pixel_batches = entry.train_batches.clone();
                        val_pixel_batches = entry.val_batches.clone();
                        test_pixel_batches = entry.test_batches.clone();
                    }
                }

                eprintln!(
                    ">>> Split: {} train batches, {} val batches, {} test batches (ratio {}/{}/{})",
                    train_pixel_batches.len(),
                    val_pixel_batches.len(),
                    test_pixel_batches.len(),
                    train_split,
                    val_split,
                    test_split
                );

                // Build train tensor batches
                let mut train_batches = Vec::with_capacity(train_pixel_batches.len());
                for chunk in &train_pixel_batches {
                    let mut current_inputs = Vec::with_capacity(chunk.len());
                    let mut current_targets = Vec::with_capacity(chunk.len());

                    for (pixels, label_id) in chunk {
                        let img_tensor_1d =
                            Tensor::<Backend, 1>::from_floats(pixels.as_slice(), &device);
                        let img_tensor_4d = img_tensor_1d.reshape([
                            1,
                            channels,
                            target_h as usize,
                            target_w as usize,
                        ]);
                        let target_tensor =
                            Tensor::<Backend, 2>::from_data([[*label_id as f32]], &device);

                        current_inputs.push(img_tensor_4d);
                        current_targets.push(target_tensor);
                    }

                    let batch = crate::entities::DynamicBatch {
                        inputs: vec![crate::entities::DynamicTensor::Dim4(Tensor::cat(
                            current_inputs,
                            0,
                        ))],
                        targets: vec![crate::entities::DynamicTensor::Dim2(Tensor::cat(
                            current_targets,
                            0,
                        ))],
                    };
                    train_batches.push(batch);
                }

                // Build val tensor batches
                let mut val_batches = Vec::with_capacity(val_pixel_batches.len());
                for chunk in &val_pixel_batches {
                    let mut current_inputs = Vec::with_capacity(chunk.len());
                    let mut current_targets = Vec::with_capacity(chunk.len());

                    for (pixels, label_id) in chunk {
                        let img_tensor_1d =
                            Tensor::<Backend, 1>::from_floats(pixels.as_slice(), &device);
                        let img_tensor_4d = img_tensor_1d.reshape([
                            1,
                            channels,
                            target_h as usize,
                            target_w as usize,
                        ]);
                        let target_tensor =
                            Tensor::<Backend, 2>::from_data([[*label_id as f32]], &device);

                        current_inputs.push(img_tensor_4d);
                        current_targets.push(target_tensor);
                    }

                    let batch = crate::entities::DynamicBatch {
                        inputs: vec![crate::entities::DynamicTensor::Dim4(Tensor::cat(
                            current_inputs,
                            0,
                        ))],
                        targets: vec![crate::entities::DynamicTensor::Dim2(Tensor::cat(
                            current_targets,
                            0,
                        ))],
                    };
                    val_batches.push(batch);
                }

                // Build test tensor batches
                let mut test_batches = Vec::with_capacity(test_pixel_batches.len());
                for chunk in &test_pixel_batches {
                    let mut current_inputs = Vec::with_capacity(chunk.len());
                    let mut current_targets = Vec::with_capacity(chunk.len());

                    for (pixels, label_id) in chunk {
                        let img_tensor_1d =
                            Tensor::<Backend, 1>::from_floats(pixels.as_slice(), &device);
                        let img_tensor_4d = img_tensor_1d.reshape([
                            1,
                            channels,
                            target_h as usize,
                            target_w as usize,
                        ]);
                        let target_tensor =
                            Tensor::<Backend, 2>::from_data([[*label_id as f32]], &device);

                        current_inputs.push(img_tensor_4d);
                        current_targets.push(target_tensor);
                    }

                    let batch = crate::entities::DynamicBatch {
                        inputs: vec![crate::entities::DynamicTensor::Dim4(Tensor::cat(
                            current_inputs,
                            0,
                        ))],
                        targets: vec![crate::entities::DynamicTensor::Dim2(Tensor::cat(
                            current_targets,
                            0,
                        ))],
                    };
                    test_batches.push(batch);
                }

                if train_batches.is_empty() {
                    eprintln!(
                        ">>> WARNING: No training batches assembled for Genome {}",
                        i
                    );
                    results.push(EvaluationResult {
                        genome_id: format!("genome_{}", i),
                        loss: 999.0,
                        accuracy: 0.0,
                    });
                    continue;
                }

                println!(
                    ">>> Assembled {} train + {} val + {} test batches for Genome {}. Starting evaluation pass...",
                    train_batches.len(),
                    val_batches.len(),
                    test_batches.len(),
                    i
                );

                // Train on training set
                crate::entities::run_eval_pass(&mut model, &train_batches, eval_epochs, 0.001);

                crate::entities::run_validation_pass(&model, &val_batches, "Validation");

                // Evaluate fitness on test set (or val if no test batches, or train)
                let (final_loss, final_acc) = if !test_batches.is_empty() {
                    crate::entities::run_validation_pass(&model, &test_batches, "Test")
                } else if !val_batches.is_empty() {
                    eprintln!(
                        ">>> WARNING: No test batches. Using validation metrics for fitness."
                    );
                    crate::entities::run_validation_pass(&model, &val_batches, "Validation")
                } else {
                    eprintln!(
                        ">>> WARNING: No validation/test batches. Using train metrics for fitness."
                    );
                    crate::entities::run_validation_pass(&model, &train_batches, "Train")
                };

                results.push(EvaluationResult {
                    genome_id: format!("genome_{}", i),
                    loss: final_loss,
                    accuracy: final_acc,
                });
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
            load_library_genome,
            save_dataset_profiles,
            load_dataset_profiles
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
