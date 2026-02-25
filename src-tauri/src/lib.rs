use std::fs;

use rfd::AsyncFileDialog;

use burn::backend::Autodiff;
use burn::backend::NdArray;
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
    type Backend = Autodiff<NdArray>;
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
            DynamicTensor::Dim4(Tensor::<Backend, 4>::random(
                [1, shape[0], shape[1], shape[2]],
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
                [1, shape[0], shape[1], shape[2]],
                Distribution::Normal(0.0, 1.0),
                &device,
            ))
        } else {
            return Err("Unsupported output shape".to_string());
        };
        targets.push(tensor);
    }

    let batch = DynamicBatch { inputs, targets };

    println!("Starting simple training for 2 epochs...");
    train_simple(model, &[batch], 2, 0.01);

    println!("Training step completed successfully.");
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            save_genome,
            load_genome,
            test_neural_net_training
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
