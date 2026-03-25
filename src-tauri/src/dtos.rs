use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn default_relu() -> String {
    "relu".to_string()
}

#[derive(Deserialize, Clone, Debug)]
pub struct KernelSizeDto {
    pub h: u8,
    pub w: u8,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    Conv2D {
        filters: u64,
        kernel_size: KernelSizeDto,
        stride: u8,
        padding: u8,
        dilation: u8,
        use_bias: bool,
        #[serde(default = "default_relu")]
        activation: String,
    },
    Conv1D {
        filters: u64,
        kernel_size: u64,
        stride: u8,
        padding: u8,
        dilation: u8,
        use_bias: bool,
        #[serde(default = "default_relu")]
        activation: String,
    },
    Dense {
        units: u64,
        activation: String,
        use_bias: bool,
    },
    Flatten {},
    Input {
        output_shape: Vec<u64>,
    },
    Output {
        input_shape: Vec<u64>,
    },
    Pooling {
        pool_type: String,
        kernel_size: KernelSizeDto,
        stride: u8,
        padding: u8,
    },
    Add {},
    Concat {},
    Dropout {
        prob: f64,
    },
    BatchNorm {
        epsilon: f64,
        momentum: f64,
    },
    LayerNorm {
        epsilon: f64,
    },
    Dropout2D {
        prob: f64,
    },
    GaussianNoise {
        std_dev: f64,
    },
    LSTM {
        hidden_units: u64,
        gate_activation: String,
        cell_activation: String,
        hidden_activation: String,
        use_bias: bool,
    },
    GRU {
        hidden_units: u64,
        gate_activation: String,
        hidden_activation: String,
        use_bias: bool,
        reset_after: bool,
    },
    MultiHeadAttention {
        n_heads: u64,
        dropout: f64,
        quiet_softmax: bool,
    },
    TransformerEncoderBlock {
        n_heads: u64,
        d_ff: u64,
        dropout: f64,
        activation: String,
        norm_first: bool,
    },
}

// ---------------------------------------------------------------------------
// Dataset Profile DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Debug)]
pub struct DatasetProfilesRoot {
    pub state: DatasetProfileState,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DatasetProfileState {
    pub profiles: Vec<DatasetProfile>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DatasetProfile {
    pub id: String,
    pub name: String,
    pub source_path: Option<String>,
    pub streams: Vec<DataStream>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub enum DataType {
    Image,
    Vector,
    Categorical,
    Text,
    TemporalSequence,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DataStream {
    pub id: String,
    pub alias: String,
    pub role: String, // "Input" | "Target" | "Ignore"
    pub data_type: DataType,
    pub tensor_shape: Vec<usize>,
    #[serde(rename = "numClasses", default)]
    pub num_classes: Option<usize>,
    pub locator: DataLocatorDef,
    pub preprocessing: Option<PreprocessingSettings>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum DataLocatorDef {
    GlobPattern {
        pattern: String,
    },
    FolderMapping, // { "type": "FolderMapping" }
    CompanionFile {
        #[serde(rename = "pathTemplate")]
        path_template: String,
        parser: String, // "YOLO" | "Text" | "COCO_Subset"
    },
    MasterIndex {
        #[serde(rename = "indexPath")]
        index_path: String,
        #[serde(rename = "keyField")]
        key_field: String,
        #[serde(rename = "valueField")]
        value_field: String,
        #[serde(rename = "hasHeaders")]
        has_headers: bool,
    },
    CsvDataset(CsvDatasetDef),
    None,
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CsvDatasetDef {
    pub csv_path: String,
    pub has_headers: bool,
    pub sample_mode: String, // "row" | "temporal_window"
    pub feature_columns: Vec<String>,
    pub target_column: String,
    pub window_size: Option<usize>,
    pub window_stride: Option<usize>,
    pub preprocessing: CsvPreprocessingConfig,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CsvPreprocessingConfig {
    pub normalization: String, // "none" | "global" | "per-sample" | "per-channel"
    pub handle_missing: String, // "skip" | "interpolate" | "mean"
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PreprocessingSettings {
    pub vision: Option<VisionSettings>,
    pub tabular: Option<TabularSettings>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VisionSettings {
    pub resize: Vec<u32>, // e.g. [256, 256]
    pub grayscale: bool,
    pub normalization: String, // "0-1" | "imagenet" | "none"
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TabularSettings {
    pub normalization: String, // "min-max" | "z-score" | "none"
    #[serde(rename = "oneHot")]
    pub one_hot: bool,
    #[serde(rename = "fillMissing")]
    pub fill_missing: String, // "mean" | "median" | "mode" | "drop"
}

// ---------------------------------------------------------------------------
// Dataset Validation DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub component: String, // "InputShape" | "OutputShape" | "SampleAlignment" | "CSV" | "Compatibility"
    pub message: String,
    pub suggested_fix: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DatasetValidationReport {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub input_shapes: HashMap<String, Vec<usize>>, // stream_id -> shape
    pub output_shape: Option<Vec<usize>>,
    pub total_valid_samples: usize,
    pub can_start_evolution: bool,
}

// ---------------------------------------------------------------------------
// Zero-Cost Proxy Configuration DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ZeroCostConfigDto {
    pub enabled: bool,
    pub strategy: String, // "two-stage" | "early-stopping"
    #[serde(rename = "fastPassThreshold")]
    pub fast_pass_threshold: f32,
    #[serde(rename = "partialTrainingEpochs")]
    pub partial_training_epochs: u32,
    #[serde(rename = "useVoting")]
    pub use_voting: bool,
}

// ---------------------------------------------------------------------------
// Performance Profiling DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrainingProfiler {
    pub train_start_ms: u64,
    pub first_batch_ms: u64,
    pub train_end_ms: u64,
    pub total_train_duration_ms: u64,
    pub val_start_ms: u64,
    pub val_end_ms: u64,
    pub val_duration_ms: u64,
    pub test_start_ms: u64,
    pub test_end_ms: u64,
    pub test_duration_ms: u64,
    pub peak_active_memory_mb: f32,
    pub peak_model_params_mb: f32,
    pub peak_gradient_mb: f32,
    pub peak_optim_state_mb: f32,
    pub peak_activation_mb: f32,
    pub samples_per_sec: f32,
    pub inference_msec_per_sample: f32,
    pub batch_count: u32,
    pub early_stop_epoch: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenerationProfilingStats {
    pub generation_number: u32,
    pub total_training_ms: u64,
    pub total_inference_ms: u64,
    pub avg_samples_per_sec: f32,
    pub peak_concurrent_vram_mb: f32,
    pub total_jobs_completed: u32,
    pub total_jobs_failed: u32,
}

// ---------------------------------------------------------------------------
// Multi-Objective Optimization DTOs (Pareto)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenomeObjectives {
    pub genome_id: String,
    pub accuracy: f32,
    pub inference_latency_ms: f32,
    pub model_size_mb: f32,
    pub training_time_ms: u64,
    pub is_dominated: bool,
    pub domination_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenerationParetoFront {
    pub generation: u32,
    pub total_genomes: u32,
    pub pareto_members: Vec<GenomeObjectives>,
    pub objectives_3d: Vec<(f32, f32, f32)>, // (accuracy, latency, size)
}

// ---------------------------------------------------------------------------
// Device Profile DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComputeType {
    Arm,
    X86,
    Gpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceProfile {
    pub device_id: String,
    pub device_name: String,
    pub compute_capability: ComputeType,
    pub ram_mb: u32,
    pub vram_mb: Option<u32>,
    pub inference_latency_budget_ms: f32,
    pub training_available: bool,
    pub power_budget_mw: Option<u32>,
    pub max_model_size_mb: Option<f32>,
    pub target_fps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceTemplateDto {
    pub id: String,
    pub name: String,
    pub constraints: crate::device_profiles::DeviceResourceConstraints,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreateDeviceTemplateInput {
    pub name: String,
    pub constraints: crate::device_profiles::DeviceResourceConstraints,
    pub notes: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct UpdateDeviceTemplatePatch {
    pub name: Option<String>,
    pub constraints: Option<crate::device_profiles::DeviceResourceConstraints>,
    pub notes: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DuplicateDeviceTemplateInput {
    pub id: String,
    pub new_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceLibraryImportMode {
    Merge,
    Replace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ImportDeviceLibraryInput {
    pub path: String,
    pub mode: DeviceLibraryImportMode,
}

// ---------------------------------------------------------------------------
// Genealogy Tracking DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
pub enum MutationType {
    Random,
    AddNode {
        node_type: String,
        source: String,
        target: String,
    },
    RemoveNode {
        node_id: String,
    },
    RemoveSubgraph {
        node_ids: Vec<String>,
    },
    ParameterMutation {
        layer_id: String,
        param_name: String,
    },
    ParameterScale {
        layer_id: String,
        scale_factor: f32,
    },
    Crossover {
        parent1: String,
        parent2: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenomeGeneology {
    pub genome_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,
    pub mutation_type: MutationType,
    pub mutation_params: serde_json::Value,
    pub fitness: f32,
    pub accuracy: f32,
    pub created_at_ms: u64,
}

// ---------------------------------------------------------------------------
// Stopping Criteria DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum StoppingCriterion {
    GenerationLimit {
        max_generations: u32,
    },
    FitnessPlateau {
        patience_generations: u32,
        improvement_threshold: f32,
        monitor: String, // "best_fitness" | "pareto_coverage" | "population_avg"
    },
    TimeLimit {
        max_seconds: u32,
    },
    TargetAccuracy {
        threshold: f32,
    },
    ManualStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct StoppingPolicy {
    pub criteria: Vec<StoppingCriterion>,
    pub policy_type: String, // "any" | "all"
}

// ---------------------------------------------------------------------------
// Training Event DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "data")]
pub enum TrainingEvent {
    RunStarted {
        run_id: String,
    },
    JobZeroCostScored {
        run_id: String,
        job_id: String,
        genome_id: String,
        proxy_score: f32,
        strategy_decision: String,
    },
    JobStarted {
        run_id: String,
        job_id: String,
        genome_id: String,
    },
    JobProgress {
        run_id: String,
        job_id: String,
        epoch: u32,
        batch: u32,
        loss: f32,
    },
    JobFinished {
        run_id: String,
        job_id: String,
        genome_id: String,
        result: EvaluationResult,
    },
    GenerationParetoComputed {
        run_id: String,
        generation: u32,
        pareto_front: GenerationParetoFront,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvaluationResult {
    pub genome_id: String,
    pub loss: f32,
    pub accuracy: f32,
}

// ---------------------------------------------------------------------------
// Extended Training Result with Profiling & Genealogy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrainingResult {
    pub genome_id: String,
    pub loss: f32,
    pub accuracy: f32,
    pub profiler: Option<TrainingProfiler>,
    pub objectives: Option<GenomeObjectives>,
    pub genealogy: Option<GenomeGeneology>,
}

// ---------------------------------------------------------------------------
// Training Job DTOs (for Orchestrator)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrainingJob {
    pub job_id: String,
    pub run_id: String,
    pub genome_id: String,
    pub genome_json: String,
    pub training_params: serde_json::Value,
    pub estimated_vram_mb: u64,
    pub dataset_name: String,
    pub priority: i32,
    pub created_at_ms: u64,
    pub proxy_decision: Option<String>,
}

// ---------------------------------------------------------------------------
// Run State DTO (for Orchestrator)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RunState {
    pub run_id: String,
    pub status: String,
    pub created_at_ms: u64,
    pub started_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
    pub queued_jobs: Vec<String>,
    pub active_jobs: Vec<String>,
    pub completed_jobs: Vec<String>,
    pub failed_jobs: Vec<String>,
    pub max_parallel_jobs: u32,
}

// ---------------------------------------------------------------------------
// Genome Library Entry with Extended Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RustGenomeLibraryEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub genome_json: String,
    pub created_at: String,
    pub is_pinned: bool,
    pub is_hidden: bool,
    pub source_generation: Option<u32>,
    pub parent_genomes: Vec<String>,
    pub fitness_metrics: Option<GenomeObjectives>,
    pub profiler_data: Option<TrainingProfiler>,
    pub model_weights: Option<String>,
    pub device_profile_target: Option<DeviceProfile>,
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_serialization() {
        let profiler = TrainingProfiler {
            train_start_ms: 1000,
            first_batch_ms: 1050,
            train_end_ms: 2000,
            total_train_duration_ms: 1000,
            val_start_ms: 2000,
            val_end_ms: 2100,
            val_duration_ms: 100,
            test_start_ms: 2100,
            test_end_ms: 2200,
            test_duration_ms: 100,
            peak_active_memory_mb: 1024.5,
            peak_model_params_mb: 512.0,
            peak_gradient_mb: 256.0,
            peak_optim_state_mb: 256.0,
            peak_activation_mb: 512.0,
            samples_per_sec: 100.0,
            inference_msec_per_sample: 10.0,
            batch_count: 100,
            early_stop_epoch: None,
        };

        let json = serde_json::to_string(&profiler).unwrap();
        let deserialized: TrainingProfiler = serde_json::from_str(&json).unwrap();
        assert_eq!(profiler.train_start_ms, deserialized.train_start_ms);
        assert_eq!(profiler.peak_active_memory_mb, deserialized.peak_active_memory_mb);
    }

    #[test]
    fn test_generation_profiling_stats_serialization() {
        let stats = GenerationProfilingStats {
            generation_number: 5,
            total_training_ms: 50000,
            total_inference_ms: 5000,
            avg_samples_per_sec: 100.0,
            peak_concurrent_vram_mb: 8192.0,
            total_jobs_completed: 32,
            total_jobs_failed: 2,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: GenerationProfilingStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.generation_number, deserialized.generation_number);
        assert_eq!(stats.total_jobs_completed, deserialized.total_jobs_completed);
    }

    #[test]
    fn test_genome_objectives_serialization() {
        let obj = GenomeObjectives {
            genome_id: "test-1".to_string(),
            accuracy: 0.95,
            inference_latency_ms: 50.0,
            model_size_mb: 10.0,
            training_time_ms: 5000,
            is_dominated: false,
            domination_count: 0,
        };

        let json = serde_json::to_string(&obj).unwrap();
        let deserialized: GenomeObjectives = serde_json::from_str(&json).unwrap();
        assert_eq!(obj.accuracy, deserialized.accuracy);
        assert_eq!(obj.genome_id, deserialized.genome_id);
        assert!(!deserialized.is_dominated);
    }

    #[test]
    fn test_generation_pareto_front_serialization() {
        let members = vec![
            GenomeObjectives {
                genome_id: "g1".to_string(),
                accuracy: 0.95,
                inference_latency_ms: 50.0,
                model_size_mb: 10.0,
                training_time_ms: 5000,
                is_dominated: false,
                domination_count: 0,
            },
            GenomeObjectives {
                genome_id: "g2".to_string(),
                accuracy: 0.90,
                inference_latency_ms: 40.0,
                model_size_mb: 8.0,
                training_time_ms: 4000,
                is_dominated: false,
                domination_count: 0,
            },
        ];

        let pareto = GenerationParetoFront {
            generation: 10,
            total_genomes: 100,
            pareto_members: members.clone(),
            objectives_3d: vec![(0.95, 50.0, 10.0), (0.90, 40.0, 8.0)],
        };

        let json = serde_json::to_string(&pareto).unwrap();
        let deserialized: GenerationParetoFront = serde_json::from_str(&json).unwrap();
        assert_eq!(pareto.generation, deserialized.generation);
        assert_eq!(pareto.pareto_members.len(), deserialized.pareto_members.len());
    }

    #[test]
    fn test_compute_type_serialization() {
        let types = vec![ComputeType::Arm, ComputeType::X86, ComputeType::Gpu];

        for ct in types {
            let json = serde_json::to_string(&ct).unwrap();
            let deserialized: ComputeType = serde_json::from_str(&json).unwrap();
            assert_eq!(ct, deserialized);
        }
    }

    #[test]
    fn test_device_profile_serialization() {
        let device = DeviceProfile {
            device_id: "dev-001".to_string(),
            device_name: "ARM Cortex-A72".to_string(),
            compute_capability: ComputeType::Arm,
            ram_mb: 4096,
            vram_mb: None,
            inference_latency_budget_ms: 100.0,
            training_available: false,
            power_budget_mw: Some(5),
            max_model_size_mb: Some(50.0),
            target_fps: Some(30.0),
        };

        let json = serde_json::to_string(&device).unwrap();
        let deserialized: DeviceProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(device.device_id, deserialized.device_id);
        assert_eq!(device.compute_capability, deserialized.compute_capability);
        assert_eq!(device.ram_mb, deserialized.ram_mb);
    }

    #[test]
    fn test_mutation_type_serialization() {
        let mutations = vec![
            MutationType::Random,
            MutationType::AddNode {
                node_type: "Dense".to_string(),
                source: "n1".to_string(),
                target: "n2".to_string(),
            },
            MutationType::RemoveNode {
                node_id: "n1".to_string(),
            },
            MutationType::Crossover {
                parent1: "g1".to_string(),
                parent2: "g2".to_string(),
            },
        ];

        for m in mutations {
            let json = serde_json::to_string(&m).unwrap();
            let deserialized: MutationType = serde_json::from_str(&json).unwrap();
            assert_eq!(m, deserialized);
        }
    }

    #[test]
    fn test_genome_genealogy_serialization() {
        let genealogy = GenomeGeneology {
            genome_id: "genome-123".to_string(),
            generation: 42,
            parent_ids: vec!["parent-1".to_string(), "parent-2".to_string()],
            mutation_type: MutationType::Random,
            mutation_params: serde_json::json!({ "mutation_rate": 0.05 }),
            fitness: 0.92,
            accuracy: 0.95,
            created_at_ms: 1704067200000,
        };

        let json = serde_json::to_string(&genealogy).unwrap();
        let deserialized: GenomeGeneology = serde_json::from_str(&json).unwrap();
        assert_eq!(genealogy.genome_id, deserialized.genome_id);
        assert_eq!(genealogy.generation, deserialized.generation);
        assert_eq!(genealogy.parent_ids.len(), deserialized.parent_ids.len());
    }

    #[test]
    fn test_stopping_criterion_serialization() {
        let criteria = vec![
            StoppingCriterion::GenerationLimit {
                max_generations: 100,
            },
            StoppingCriterion::FitnessPlateau {
                patience_generations: 5,
                improvement_threshold: 0.01,
                monitor: "best_fitness".to_string(),
            },
            StoppingCriterion::TimeLimit { max_seconds: 3600 },
            StoppingCriterion::TargetAccuracy { threshold: 0.99 },
            StoppingCriterion::ManualStop,
        ];

        for criterion in criteria {
            let json = serde_json::to_string(&criterion).unwrap();
            let _deserialized: StoppingCriterion = serde_json::from_str(&json).unwrap();
            // Due to enum variants, just verify it deserializes successfully
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_stopping_policy_serialization() {
        let policy = StoppingPolicy {
            criteria: vec![
                StoppingCriterion::GenerationLimit {
                    max_generations: 100,
                },
                StoppingCriterion::TargetAccuracy { threshold: 0.95 },
            ],
            policy_type: "any".to_string(),
        };

        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: StoppingPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(policy.policy_type, deserialized.policy_type);
        assert_eq!(policy.criteria.len(), deserialized.criteria.len());
    }

    #[test]
    fn test_training_event_serialization() {
        let events = vec![
            TrainingEvent::RunStarted {
                run_id: "run-001".to_string(),
            },
            TrainingEvent::JobProgress {
                run_id: "run-001".to_string(),
                job_id: "job-001".to_string(),
                epoch: 5,
                batch: 32,
                loss: 0.45,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let _deserialized: TrainingEvent = serde_json::from_str(&json).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_evaluation_result_serialization() {
        let result = EvaluationResult {
            genome_id: "genome-456".to_string(),
            loss: 0.23,
            accuracy: 0.96,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: EvaluationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.genome_id, deserialized.genome_id);
        assert_eq!(result.accuracy, deserialized.accuracy);
    }

    #[test]
    fn test_training_result_with_profiling() {
        let profiler = Some(TrainingProfiler {
            train_start_ms: 1000,
            first_batch_ms: 1050,
            train_end_ms: 2000,
            total_train_duration_ms: 1000,
            val_start_ms: 2000,
            val_end_ms: 2100,
            val_duration_ms: 100,
            test_start_ms: 2100,
            test_end_ms: 2200,
            test_duration_ms: 100,
            peak_active_memory_mb: 1024.5,
            peak_model_params_mb: 512.0,
            peak_gradient_mb: 256.0,
            peak_optim_state_mb: 256.0,
            peak_activation_mb: 512.0,
            samples_per_sec: 100.0,
            inference_msec_per_sample: 10.0,
            batch_count: 100,
            early_stop_epoch: Some(80),
        });

        let result = TrainingResult {
            genome_id: "genome-789".to_string(),
            loss: 0.15,
            accuracy: 0.98,
            profiler,
            objectives: None,
            genealogy: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: TrainingResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.genome_id, deserialized.genome_id);
        assert!(deserialized.profiler.is_some());
    }

    #[test]
    fn test_genome_library_entry_with_metadata() {
        let entry = RustGenomeLibraryEntry {
            id: "lib-entry-001".to_string(),
            name: "High Accuracy Model".to_string(),
            description: "Neural network optimized for accuracy".to_string(),
            genome_json: r#"{"nodes": [], "connections": []}"#.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            is_pinned: true,
            is_hidden: false,
            source_generation: Some(50),
            parent_genomes: vec!["parent-g1".to_string()],
            fitness_metrics: Some(GenomeObjectives {
                genome_id: "lib-entry-001".to_string(),
                accuracy: 0.97,
                inference_latency_ms: 60.0,
                model_size_mb: 15.0,
                training_time_ms: 8000,
                is_dominated: false,
                domination_count: 0,
            }),
            profiler_data: None,
            model_weights: Some("weights".to_string()),
            device_profile_target: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: RustGenomeLibraryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry.id, deserialized.id);
        assert_eq!(entry.is_pinned, deserialized.is_pinned);
        assert!(deserialized.fitness_metrics.is_some());
    }
}
