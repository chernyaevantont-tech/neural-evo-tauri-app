# Backend Rust Implementation Guide — Conv1D, LSTM, GRU, Attention

## Status Update

✅ **Frontend TypeScript** - 100% complete
- Conv1D, LSTM, GRU nodes fully implemented with tests
- MultiHeadAttention, TransformerEncoderBlock nodes fully implemented with tests
- **All 92 tests passing**

🚧 **Backend Rust** - Implementation template provided below

---

## Changes Required to `src-tauri/src/entities.rs`

### 1. Add Burn Imports (top of file)

```rust
use burn::nn::{
    // ... existing imports ...
    conv::Conv1d, // Add this
    Conv1dConfig, // Add this
    lstm::{ Lstm, LstmConfig, LstmState }, // Add these
    gru::{ Gru, GruConfig }, // Add these
    attention::{ MultiHeadAttention, MultiHeadAttentionConfig, MhaInput, MhaOutput }, // Add these
    transformer::{ TransformerEncoderLayer, TransformerEncoderConfig }, // Add these
};
```

### 2. Update Operation Enum

Add these variants to the `Operation` enum:

```rust
#[derive(Clone, Debug)]
pub enum Operation {
    // ... existing variants ...
    
    /// 1D Convolution for sequences
    Conv1D {
        conv1d_idx: usize,
        activation: String,
    },
    
    /// LSTM layer
    LSTM {
        lstm_idx: usize,
    },
    
    /// GRU layer
    GRU {
        gru_idx: usize,
    },
    
    /// Multi-head Self-Attention
    MultiHeadAttention {
        mha_idx: usize,
    },
    
    /// Transformer Encoder Block (MHA + FFN)
    TransformerEncoderBlock {
        transformer_idx: usize,
    },
}
```

### 3. Update GraphModel Struct

Add these fields to the `GraphModel` struct:

```rust
#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    // ... existing fields ...
    
    /// 1D Convolution layers
    pub conv1ds: Vec<Conv1d<B>>,
    
    /// LSTM layers (store hidden_state_size for reshaping)
    pub lstms: Vec<Lstm<B>>,
    pub lstm_hidden_sizes: Vec<usize>, // Track hidden dims for shape calculations
    
    /// GRU layers
    pub grus: Vec<Gru<B>>,
    pub gru_hidden_sizes: Vec<usize>,
    
    /// Multi-head attention
    pub mha_layers: Vec<MultiHeadAttention<B>>,
    pub mha_configs: Vec<(usize, usize)>, // (d_model, n_heads) for reshaping
    
    /// Transformer encoder blocks
    pub transformer_encoders: Vec<TransformerEncoderLayer<B>>,
    pub transformer_configs: Vec<TransformerEncoderConfig>,
    
    // ... rest of existing fields ...
    pub execution_plan: Ignored<Vec<Instruction>>,
}
```

### 4. Implement Conv1D Build Logic

In the `GraphModel::build()` match statement, add:

```rust
NodeDtoJSON::Conv1D {
    filters,
    kernel_size,
    stride,
    padding,
    dilation,
    use_bias,
    activation,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let in_channels = prev_shape[0];
    let seq_len = prev_shape[1];
    
    let mut actual_filters = *filters as usize;
    if let Some(&out_idx) = connects_to_output.get(&node_id) {
        if let Some(overrides) = output_shape_overrides {
            if let Some(ov) = overrides.get(out_idx) {
                if !ov.is_empty() {
                    actual_filters = ov[0];
                }
            }
        }
    }

    let conv1d = Conv1dConfig::new(in_channels, actual_filters, *kernel_size as usize)
        .with_stride(*stride as usize)
        .with_padding(*padding as usize)
        .with_dilation(*dilation as usize)
        .with_bias(*use_bias)
        .init(device);

    let conv1d_idx = conv1ds.len();
    conv1ds.push(conv1d);

    // Output length: floor((seq_len + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    let seq_out = (seq_len + 2 * (*padding as usize)
        - (*dilation as usize) * (*kernel_size as usize - 1)
        - 1) / (*stride as usize)
        + 1;

    (
        Operation::Conv1D {
            conv1d_idx,
            activation: activation.clone(),
        },
        vec![actual_filters, seq_out],
    )
}
```

### 5. Implement LSTM Build Logic

```rust
NodeDtoJSON::LSTM {
    hidden_units,
    gate_activation,
    cell_activation,
    hidden_activation,
    use_bias,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let input_size = prev_shape[0];
    let seq_len = prev_shape[1];
    let hidden_units = *hidden_units as usize;

    let lstm = LstmConfig::new(input_size, hidden_units, *use_bias)
        .with_gate_activation(ActivationConfig::from_string(gate_activation))
        .with_cell_activation(ActivationConfig::from_string(cell_activation))
        .with_hidden_activation(ActivationConfig::from_string(hidden_activation))
        .init(device);

    let lstm_idx = lstms.len();
    lstms.push(lstm);
    lstm_hidden_sizes.push(hidden_units);

    // LSTM output: [seq_len, hidden_units]
    (
        Operation::LSTM { lstm_idx },
        vec![hidden_units, seq_len],
    )
}
```

### 6. Implement GRU Build Logic

```rust
NodeDtoJSON::GRU {
    hidden_units,
    gate_activation,
    hidden_activation,
    use_bias,
    reset_after,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let input_size = prev_shape[0];
    let seq_len = prev_shape[1];
    let hidden_units = *hidden_units as usize;

    let gru = GruConfig::new(input_size, hidden_units, *use_bias)
        .with_gate_activation(ActivationConfig::from_string(gate_activation))
        .with_hidden_activation(ActivationConfig::from_string(hidden_activation))
        .with_reset_after(*reset_after)
        .init(device);

    let gru_idx = grus.len();
    grus.push(gru);
    gru_hidden_sizes.push(hidden_units);

    // GRU output: [seq_len, hidden_units]
    (
        Operation::GRU { gru_idx },
        vec![hidden_units, seq_len],
    )
}
```

### 7. Implement MultiHeadAttention Build Logic

```rust
NodeDtoJSON::MultiHeadAttention {
    n_heads,
    dropout,
    quiet_softmax,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let d_model = prev_shape[0];
    let seq_len = prev_shape[1];

    let mha = MultiHeadAttentionConfig::new(d_model, *n_heads as usize)
        .with_dropout(*dropout)
        .with_quiet_softmax(*quiet_softmax)
        .init(device);

    let mha_idx = mha_layers.len();
    mha_layers.push(mha);
    mha_configs.push((d_model, *n_heads as usize));

    // MHA preserves shape
    (
        Operation::MultiHeadAttention { mha_idx },
        vec![d_model, seq_len],
    )
}
```

### 8. Implement TransformerEncoderBlock Build Logic

```rust
NodeDtoJSON::TransformerEncoderBlock {
    n_heads,
    d_ff,
    dropout,
    activation,
    norm_first,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let d_model = prev_shape[0];
    let seq_len = prev_shape[1];

    let config = TransformerEncoderConfig {
        d_model,
        d_ff: *d_ff as usize,
        n_heads: *n_heads as usize,
        n_layers: 1,
        dropout: *dropout,
        norm_first: *norm_first,
        activation: ActivationConfig::from_string(activation),
        initializer: Initializer::default(),
        layer_norm_eps: 1e-6,
        quiet_softmax: false,
    };

    let encoder = config.init(device);

    let transformer_idx = transformer_encoders.len();
    transformer_encoders.push(encoder);
    transformer_configs.push(config);

    // Transformer preserves shape
    (
        Operation::TransformerEncoderBlock { transformer_idx },
        vec![d_model, seq_len],
    )
}
```

---

## Forward Pass Implementation

### 1. Update GraphModel::forward() Method

In the main forward loop, add handling for new operation types:

```rust
impl<B: AutodiffBackend> SupervisedStep<DynamicBatch<B>, DynamicOutput<B>> for GraphModel<B> {
    fn step(&self, batch: DynamicBatch<B>) -> TrainOutput<DynamicOutput<B>> {
        // ... existing setup code ...

        for instr in &self.execution_plan {
            match &instr.op {
                // ... existing operations ...

                Operation::Conv1D {
                    conv1d_idx,
                    activation,
                } => {
                    let input = &tensors[instr.input_ids[0]];
                    let input_2d = match input {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => panic!("Conv1D expects 2D input [seq_len, channels]"),
                    };

                    // Input shape: [seq_len, input_channels]
                    // Reshape to [1, input_channels, seq_len] for Burn Conv1d
                    let [seq_len, in_channels] = input_2d.dims();
                    let batched = input_2d.transpose().unsqueeze::<1>(0); // [1, in_channels, seq_len]

                    let output = self.conv1ds[*conv1d_idx].forward(batched);

                    // Output shape: [1, filters, seq_len_out]
                    let [_, filters, seq_len_out] = output.dims();
                    
                    // Reshape back to [seq_len_out, filters]
                    let output_2d = output.squeeze::<3>(0).transpose();

                    tensors[instr.node_id] = DynamicTensor::Dim2(output_2d);
                }

                Operation::LSTM { lstm_idx } => {
                    let input = &tensors[instr.input_ids[0]];
                    let input_2d = match input {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => panic!("LSTM expects 2D input [input_size, seq_len]"),
                    };

                    // Frontend sends [input_size, seq_len]
                    // Reshape to [1, seq_len, input_size] for Burn LSTM
                    let [input_size, seq_len] = input_2d.dims();
                    let input_3d = input_2d
                        .transpose()
                        .unsqueeze::<1>(0); // [1, seq_len, input_size]

                    // Forward through LSTM
                    let lstm = &self.lstms[*lstm_idx];
                    let (output, _state) = lstm.forward(input_3d, None);

                    // Output shape: [1, seq_len, hidden_units]
                    let hidden_size = self.lstm_hidden_sizes[*lstm_idx];
                    let [_, seq_out, _] = output.dims();

                    // Reshape back to [hidden_units, seq_len]
                    let output_2d = output
                        .squeeze::<1>(0) // [seq_len, hidden_units]
                        .transpose(); // [hidden_units, seq_len]

                    tensors[instr.node_id] = DynamicTensor::Dim2(output_2d);
                }

                Operation::GRU { gru_idx } => {
                    let input = &tensors[instr.input_ids[0]];
                    let input_2d = match input {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => panic!("GRU expects 2D input [input_size, seq_len]"),
                    };

                    let [input_size, seq_len] = input_2d.dims();
                    let input_3d = input_2d
                        .transpose()
                        .unsqueeze::<1>(0); // [1, seq_len, input_size]

                    // Forward through GRU
                    let gru = &self.grus[*gru_idx];
                    let (output, _state) = gru.forward(input_3d, None);

                    // Output: [1, seq_len, hidden_units]
                    let hidden_size = self.gru_hidden_sizes[*gru_idx];

                    // Reshape back to [hidden_units, seq_len]
                    let output_2d = output
                        .squeeze::<1>(0) // [seq_len, hidden_units]
                        .transpose(); // [hidden_units, seq_len]

                    tensors[instr.node_id] = DynamicTensor::Dim2(output_2d);
                }

                Operation::MultiHeadAttention { mha_idx } => {
                    let input = &tensors[instr.input_ids[0]];
                    let input_2d = match input {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => panic!("MHA expects 2D input [d_model, seq_len]"),
                    };

                    let [d_model, seq_len] = input_2d.dims();
                    
                    // Reshape to [1, seq_len, d_model] for Burn MHA
                    let input_3d = input_2d
                        .transpose()
                        .unsqueeze::<1>(0); // [1, seq_len, d_model]

                    // Forward: self-attention (query = key = value = input)
                    let mha_input = MhaInput::self_attn(input_3d);
                    let mha_output = self.mha_layers[*mha_idx].forward(mha_input);

                    // Output: [1, seq_len, d_model]
                    // Reshape back to [d_model, seq_len]
                    let output_2d = mha_output.context
                        .squeeze::<1>(0) // [seq_len, d_model]
                        .transpose(); // [d_model, seq_len]

                    tensors[instr.node_id] = DynamicTensor::Dim2(output_2d);
                }

                Operation::TransformerEncoderBlock { transformer_idx } => {
                    let input = &tensors[instr.input_ids[0]];
                    let input_2d = match input {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => panic!("TransformerBlock expects 2D input [d_model, seq_len]"),
                    };

                    let [d_model, seq_len] = input_2d.dims();

                    // Reshape to [1, seq_len, d_model]
                    let input_3d = input_2d
                        .transpose()
                        .unsqueeze::<1>(0); // [1, seq_len, d_model]

                    // Forward through transformer encoder layer
                    let encoder_input = TransformerEncoderInput::new(input_3d);
                    let output = self.transformer_encoders[*transformer_idx]
                        .forward(encoder_input);

                    // Output: [1, seq_len, d_model]
                    // Reshape back to [d_model, seq_len]
                    let output_2d = output
                        .squeeze::<1>(0) // [seq_len, d_model]
                        .transpose(); // [d_model, seq_len]

                    tensors[instr.node_id] = DynamicTensor::Dim2(output_2d);
                }

                // ... rest of operations ...
            }
        }

        // ... rest of implementation ...
    }
}
```

---

## Key Implementation Notes

### Shape Handling Strategy

**Critical**: Frontend sends 2D tensors, Burn modules expect 3D:

| Layer | Frontend Format | Burn Format | Reshape |
|-------|-----------------|-------------|---------|
| Conv1D | [seq_len, channels] | [1, channels, seq_len] | transpose → unsqueeze → Conv1D → squeeze → transpose |
| LSTM | [input_size, seq_len] | [1, seq_len, input_size] | transpose → unsqueeze → LSTM → squeeze → transpose |
| GRU | [input_size, seq_len] | [1, seq_len, input_size] | transpose → unsqueeze → GRU → squeeze → transpose |
| MHA | [d_model, seq_len] | [1, seq_len, d_model] | transpose → unsqueeze → MHA → squeeze → transpose |
| Transformer | [d_model, seq_len] | [1, seq_len, d_model] | transpose → unsqueeze → forward → squeeze → transpose |

### Activation Handling

Create helper function for string → Burn activation config:

```rust
fn activation_from_string(name: &str) -> ActivationConfig {
    match name {
        "sigmoid" => ActivationConfig::Sigmoid,
        "tanh" => ActivationConfig::Tanh,
        "relu" => ActivationConfig::Relu,
        "leaky_relu" => ActivationConfig::LeakyRelu(LeakyReluConfig::default()),
        "gelu" => ActivationConfig::Gelu,
        "swish" => ActivationConfig::Swish,
        _ => ActivationConfig::Relu,
    }
}
```

---

## Testing Checklist

- [ ] DTOs compile without errors
- [ ] Conv1D forward pass matches NumPy calculations
- [ ] LSTM output shape [batch=1, seq_len, hidden_units]
- [ ] GRU output matches LSTM interface
- [ ] MHA with seq_len < 1024 doesn't panic on memory
- [ ] TransformerBlock residual connections work correctly
- [ ] End-to-end genome serialization → Rust compilation → inference

---

## Next Steps

1. **Apply DTO changes** from this guide to `src-tauri/src/dtos.rs` ✅
2. **Add Burn imports** to `entities.rs` top
3. **Update Operation enum** with new variants
4. **Update GraphModel struct** with new module vectors
5. **Implement build logic** for each node type (use code blocks above)
6. **Implement forward logic** in SupervisedStep (use forward pass code)
7. **Test compilation**: `cargo build --release`
8. **Integration test**: Create genome with Conv1D → LSTM → MHA → Output, train it

