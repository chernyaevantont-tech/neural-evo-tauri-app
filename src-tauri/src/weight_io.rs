use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use serde::{Deserialize, Serialize};

use crate::dtos::TrainingProfiler;
use crate::entities::GraphModel;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportObjectives {
    pub accuracy: Option<f32>,
    pub inference_latency_ms: Option<f32>,
    pub model_size_mb: Option<f32>,
    pub train_duration_ms: Option<u64>,
    pub device_profile_id: Option<String>,
    pub lineage: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightExportMetadata {
    pub genome_id: String,
    pub created_at: String,
    pub accuracy: Option<f32>,
    pub inference_latency_ms: Option<f32>,
    pub model_size_mb: Option<f32>,
    pub train_duration_ms: Option<u64>,
    pub device_profile_id: Option<String>,
    pub lineage: Vec<String>,
}

fn weight_file_path(genome_id: &str, dir: &Path) -> PathBuf {
    dir.join(format!("{}.mpk", genome_id))
}

fn metadata_file_path(dir: &Path) -> PathBuf {
    dir.join("metadata.json")
}

pub fn save_weights<B: Backend>(
    genome_id: &str,
    model: Option<&GraphModel<B>>,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;

    let weights_path = weight_file_path(genome_id, output_dir);
    if weights_path.exists() {
        return Ok(weights_path);
    }

    let model = model.ok_or_else(|| {
        format!(
            "No serialized weights found for genome '{}' and no trained model was provided",
            genome_id
        )
    })?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let base = output_dir.join(genome_id);
    model
        .clone()
        .save_file(base, &recorder)
        .map_err(|e| e.to_string())?;

    if !weights_path.exists() {
        return Err(format!(
            "Expected weight file '{}' was not created",
            weights_path.to_string_lossy()
        ));
    }

    Ok(weights_path)
}

pub fn load_weights(genome_id: &str, input_dir: &Path) -> Result<Option<PathBuf>, String> {
    let weights_path = weight_file_path(genome_id, input_dir);
    if weights_path.exists() {
        return Ok(Some(weights_path));
    }

    Ok(None)
}

pub fn export_with_metadata(
    genome_id: &str,
    output_dir: &Path,
    objectives: &ExportObjectives,
    profiler: Option<&TrainingProfiler>,
) -> Result<(PathBuf, PathBuf), String> {
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;

    let weights_path = match load_weights(genome_id, output_dir)? {
        Some(path) => path,
        None => {
            return Err(format!(
                "No cached trained weights found for genome '{}'. Run training/evaluation first.",
                genome_id
            ));
        }
    };

    let metadata = WeightExportMetadata {
        genome_id: genome_id.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        accuracy: objectives.accuracy,
        inference_latency_ms: objectives.inference_latency_ms,
        model_size_mb: objectives.model_size_mb,
        train_duration_ms: objectives
            .train_duration_ms
            .or_else(|| profiler.map(|p| p.total_train_duration_ms)),
        device_profile_id: objectives.device_profile_id.clone(),
        lineage: objectives.lineage.clone(),
    };

    let metadata_path = metadata_file_path(output_dir);
    let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| e.to_string())?;
    fs::write(&metadata_path, metadata_json).map_err(|e| e.to_string())?;

    Ok((weights_path, metadata_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(prefix: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("{}-{}", prefix, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn save_and_load_weights_roundtrip() {
        let dir = temp_dir("weight-io");

        type TestBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let genome = [
            r#"{"node":"Input","params":{"output_shape":[4]}}"#,
            r#"{"node":"Dense","params":{"units":3,"activation":"relu","use_bias":true}}"#,
            r#"{"node":"Output","params":{"input_shape":[3]}}"#,
            "CONNECTIONS",
            "0 1",
            "1 2",
        ]
        .join("\n");
        let model = GraphModel::<TestBackend>::build(&genome, &device, None, None);

        let saved = save_weights("genome-a", Some(&model), &dir).expect("save weights");
        assert!(saved.exists());

        let loaded = load_weights("genome-a", &dir).expect("load weights");
        assert!(loaded.is_some());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn metadata_schema_contains_required_fields() {
        let dir = temp_dir("weight-export");

        type TestBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let genome = [
            r#"{"node":"Input","params":{"output_shape":[4]}}"#,
            r#"{"node":"Dense","params":{"units":3,"activation":"relu","use_bias":true}}"#,
            r#"{"node":"Output","params":{"input_shape":[3]}}"#,
            "CONNECTIONS",
            "0 1",
            "1 2",
        ]
        .join("\n");
        let model = GraphModel::<TestBackend>::build(&genome, &device, None, None);
        save_weights("genome-b", Some(&model), &dir).expect("seed real cached weights");

        let objectives = ExportObjectives {
            accuracy: Some(0.93),
            inference_latency_ms: Some(1.7),
            model_size_mb: Some(2.4),
            train_duration_ms: Some(5_200),
            device_profile_id: Some("edge-device-a".to_string()),
            lineage: vec!["parent-1".to_string(), "parent-2".to_string()],
        };

        let (_weights_path, metadata_path) =
            export_with_metadata("genome-b", &dir, &objectives, None).expect("export with metadata");

        let json = fs::read_to_string(metadata_path).expect("read metadata");
        let val: serde_json::Value = serde_json::from_str(&json).expect("parse metadata json");

        assert_eq!(val["genome_id"], "genome-b");
        assert!(val.get("created_at").is_some());
        assert_eq!(val["accuracy"], 0.93);
        assert_eq!(val["inference_latency_ms"], 1.7);
        assert_eq!(val["model_size_mb"], 2.4);
        assert_eq!(val["train_duration_ms"], 5200);
        assert_eq!(val["device_profile_id"], "edge-device-a");
        assert!(val["lineage"].is_array());

        let _ = fs::remove_dir_all(dir);
    }
}