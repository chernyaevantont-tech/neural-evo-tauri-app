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
    #[serde(rename = "numClasses")]
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
