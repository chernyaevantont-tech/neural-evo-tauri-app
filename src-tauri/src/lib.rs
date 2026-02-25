use std::fs;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            save_genome,
            load_genome,
            test_neural_net_training,
            test_train_on_image_folder
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
