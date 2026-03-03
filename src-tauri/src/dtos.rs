use serde::Deserialize;

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

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DataStream {
    pub id: String,
    pub alias: String,
    pub role: String, // "Input" | "Target" | "Ignore"
    pub locator: DataLocatorDef,
    pub preprocessing: Option<PreprocessingSettings>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DataLocatorDef {
    FolderMapping, // { "type": "FolderMapping" }
    None,
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PreprocessingSettings {
    pub vision: Option<VisionSettings>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VisionSettings {
    pub resize: Vec<u32>, // e.g. [256, 256]
    pub grayscale: bool,
}
