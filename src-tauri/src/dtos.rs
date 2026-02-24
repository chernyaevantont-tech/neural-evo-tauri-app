use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    Conv2D {
        filters: u64,
        kernel_size: Vec<u8>,
        stride: u8,
        padding: u8,
        dilation: u8,
        use_bias: bool,
    },
    Dense {
        units: u64,
        activation: String,
        use_bias: bool,
    },
    Flatten {},
    Input {
        output_shape: Vec<u64>
    },
    Output {
        input_shape: Vec<u64>,
    },
    Pooling {
        pool_type: String,
        kernel_size: Vec<u8>,
        stride: u8,
        padding: u8,
    },
    Add {},
    Concat {},
}