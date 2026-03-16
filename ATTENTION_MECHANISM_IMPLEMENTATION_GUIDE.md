# Attention Mechanism Nodes — Implementation Guide

Complete guide for adding Multi-Head Attention and Cross-Attention nodes to the neural evolution neural network editor. Leverages **Burn 0.20+** built-in attention modules (`burn::nn::attention`, `burn::nn::transformer`).

**Time to implement**: ~4-6 hours (3 node types + tests)

---

## Table of Contents

1. [Burn Attention API Overview](#1-burn-attention-api-overview)
2. [Node Architecture](#2-node-architecture)
3. [Frontend Implementation (TypeScript)](#3-frontend-implementation-typescript)
4. [Backend Implementation (Rust/Burn)](#4-backend-implementation-rustburn)
5. [Shape Propagation Rules](#5-shape-propagation-rules)
6. [Testing Strategy](#6-testing-strategy)
7. [UI Integration](#7-ui-integration)
8. [Advanced Patterns](#8-advanced-patterns)

---

## 1. Burn Attention API Overview

### MultiHeadAttention Module

```rust
// Configuration
pub struct MultiHeadAttentionConfig {
    pub d_model: usize,           // Input/output embedding dimension
    pub n_heads: usize,           // Number of attention heads
    pub dropout: f64,             // Default: 0.1
    pub min_float: f64,           // Default: -1.0e4 (for masking)
    pub quiet_softmax: bool,      // Default: false (Gemma-style)
    pub initializer: Initializer, // Weight initialization
}

// Forward signature
pub fn forward(&self, input: MhaInput<B>) -> MhaOutput<B>

// Input structure (self-attention or cross-attention)
pub struct MhaInput<B: Backend> {
    pub query: Tensor<B, 3>,      // [batch, seq_len_q, d_model]
    pub key: Tensor<B, 3>,        // [batch, seq_len_k, d_model]
    pub value: Tensor<B, 3>,      // [batch, seq_len_k, d_model]
    pub mask_pad: Option<Tensor<B, 2, Bool>>,    // [batch, seq_len]
    pub mask_attn: Option<Tensor<B, 3, Bool>>,   // [batch, seq_len_q, seq_len_k]
}

// Output
pub struct MhaOutput<B: Backend> {
    pub context: Tensor<B, 3>,    // [batch, seq_len_q, d_model]
    pub weights: Tensor<B, 4>,    // [batch, n_heads, seq_len_q, seq_len_k]
}
```

**Key Properties**:
- ✅ Self-attention: query = key = value (same tensor)
- ✅ Cross-attention: query ≠ key = value (from different sources)
- ✅ Causal masking: supported via `mask_attn` parameter
- ✅ Padding masking: supported via `mask_pad` parameter
- ✅ Dropout: built-in, applied to attention scores
- ✅ Output projection: built-in Linear layer

### CrossAttention Module

```rust
// Similar API but explicitly separates query from key/value context
pub fn forward(
    &self,
    query: Tensor<B, 3>,         // [batch, seq_q, d_model]
    context: Tensor<B, 3>,       // [batch, seq_kv, d_model]
    mask: Option<Tensor<B, 2, Bool>>,  // [batch, seq_kv]
) -> Tensor<B, 3>               // [batch, seq_q, d_model]
```

---

## 2. Node Architecture

### 2.1 MultiHeadAttentionNode (Self-Attention)

**Purpose**: Standalone multi-head self-attention layer for sequence modeling.

**Input Shape**: `[sequence_length, d_model]` (2D)
- Where `d_model` must be divisible by `n_heads`

**Output Shape**: `[sequence_length, d_model]` (2D)
- Same output dimension as input

**Parameters**:
- `d_model` (or inferred from input shape)
- `n_heads` (typically 4, 8, 16; default 4)
- `dropout` (0.0-0.5; default 0.1)
- `quiet_softmax` (bool; default false)

**Key Insight**: Output shape is deterministic: always `[input_seq_len, d_model]`.

**Compatible Predecessors**:
- Dense layers (output flattened to 1D, reshape)
- Conv1D layers (output [seq_len, filters])
- Other MHA layers
- RNN outputs (LSTM/GRU: [seq_len, hidden])

**Compatible Successors**:
- MHA layers
- Dense
- Conv1D
- Output layer

### 2.2 CrossAttentionNode

**Purpose**: Cross-attention for encoder-decoder architectures. Attends query sequence to separate key/value context.

**Requires Special Handling**: Cross-attention nodes need TWO inputs:
1. Primary input: query sequence `[seq_q, d_model]`
2. Context input: key/value sequence `[seq_kv, d_model]` (from encoder)

**Graph Design Challenge**: Standard computational graphs are linear (single input per node). Cross-attention breaks this pattern.

**Solution Options**:

**Option A: Merge Node Pattern** (Recommended)
- Model cross-attention as a merge node (like `Concat2D`, `Add`)
- Takes two inputs: primary (query) and context (key/value)
- Output: `[seq_query, d_model]`
- Simpler integration with existing graph structure

**Option B: Context Attachment**
- Store context reference separately
- More complex but more flexible
- Useful for multi-hop attention

For this guide, **Option A (merge pattern)** is specified.

**Implementation Strategy**:
```typescript
export class CrossAttentionNode extends BaseNode {
    protected GetIsMerging(): boolean {
        return true;  // Merge node marker
    }
    
    // Takes TWO .previous edges:
    // this.previous[0] = query input
    // this.previous[1] = context input (key/value)
}
```

### 2.3 TransformerEncoderBlockNode

**Purpose**: Full transformer encoder block with self-attention + position-wise feed-forward.

**Input Shape**: `[sequence_length, d_model]`

**Output Shape**: `[sequence_length, d_model]`

**Parameters**:
- `d_model` (embedding dimension)
- `n_heads` (number of heads; 4-16)
- `d_ff` (feed-forward hidden dimension; typically 4 × d_model)
- `dropout` (0.0-0.5)
- `activation` (relu, gelu; default gelu)
- `norm_first` (pre-norm vs post-norm; default false)

**Internal Components**:
- MultiHeadAttention (self-attention)
- PositionWiseFeedForward (Dense + activation + Dense)
- LayerNorm (2×)
- Dropout (2×)
- Residual connections (2×)

**Advantage over Separate Nodes**:
- ✅ Single node encapsulates complete transformer encoder layer
- ✅ Prevents accidentally misconfiguring the layer
- ✅ Simpler than manually chaining MHA + FFNN nodes
- ✅ Built-in layer normalization and residual connections

---

## 3. Frontend Implementation (TypeScript)

### 3.1 MultiHeadAttentionNode

```typescript
import { BaseNode } from "../base_node";
import type { IShape } from "../../types";

export class MultiHeadAttentionNode extends BaseNode {
    private nHeads: number = 4;
    private dropout: number = 0.1;
    private quietSoftmax: boolean = false;

    constructor(nodeId: string, genomeId: string, shape: IShape = [512, 64]) {
        super(nodeId, genomeId, shape);
        this.nodeType = "MultiHeadAttention";
    }

    protected CalculateOutputShape(): void {
        // Output shape same as input shape: [seq_len, d_model]
        this.outputShape = this.inputShape;
    }

    GetNodeType = (): string => "MultiHeadAttention";

    GetInfo = (): string => {
        return JSON.stringify({
            node: "MultiHeadAttention",
            params: {
                n_heads: this.nHeads,
                dropout: this.dropout,
                quiet_softmax: this.quietSoftmax,
            },
        });
    };

    GetResources = (): Map<string, number> => {
        const resources = new Map<string, number>();
        const [seqLen, dModel] = this.inputShape;

        // Compute parameters for query, key, value, output projections
        // Each is a Linear(d_model, d_model)
        // Parameters = d_model * d_model + d_model (weight + bias)
        const projectionParams = dModel * dModel + dModel;
        resources.set("trainable_params", projectionParams * 4); // 4 projections

        // Memory for attention scores: [seq_len, seq_len] (float32)
        resources.set("intermediate_memory", seqLen * seqLen * 4);
        return resources;
    };

    protected Mutate(options: Map<string, number>): void {
        // N_heads: [1, min(d_model/2, 16)]
        const dModel = this.inputShape[1];
        const maxHeads = Math.min(Math.floor(dModel / 2), 16);
        if (Math.random() < 0.3) {
            this.nHeads = Math.max(1, Math.floor(Math.random() * maxHeads));
        }

        // Dropout: [0.0, 0.5]
        if (Math.random() < 0.3) {
            this.dropout = Math.random() * 0.5;
        }

        // Quiet softmax: flip with 20% probability
        if (Math.random() < 0.2) {
            this.quietSoftmax = !this.quietSoftmax;
        }
    }

    protected GetExpectedInputDimensions(): number {
        // Expects 2D input: [seq_len, d_model]
        return 2;
    }

    GetOutputDimensions(): number {
        return 2;  // Outputs [seq_len, d_model]
    }

    protected _CloneImpl(): BaseNode {
        const clone = new MultiHeadAttentionNode(this.id, this.genomeId, this.inputShape);
        clone.nHeads = this.nHeads;
        clone.dropout = this.dropout;
        clone.quietSoftmax = this.quietSoftmax;
        clone.outputShape = this.outputShape;
        return clone;
    }

    // Properties for UI editing
    SetNHeads(n: number): void {
        this.nHeads = Math.max(1, Math.min(n, 16));
        this.PropagateShapeUpdate();
    }

    GetNHeads(): number {
        return this.nHeads;
    }

    SetDropout(d: number): void {
        this.dropout = Math.max(0, Math.min(d, 1.0));
    }

    GetDropout(): number {
        return this.dropout;
    }

    SetQuietSoftmax(q: boolean): void {
        this.quietSoftmax = q;
    }

    GetQuietSoftmax(): boolean {
        return this.quietSoftmax;
    }
}
```

### 3.2 CrossAttentionNode

```typescript
import { BaseNode } from "../base_node";
import type { IShape } from "../../types";

export class CrossAttentionNode extends BaseNode {
    private nHeads: number = 4;
    private dropout: number = 0.1;
    private quietSoftmax: boolean = false;

    constructor(nodeId: string, genomeId: string, shape: IShape = [512, 64]) {
        super(nodeId, genomeId, shape);
        this.nodeType = "CrossAttention";
    }

    protected CalculateOutputShape(): void {
        // Output shape = query input shape (first previous)
        // [ seq_len_query, d_model ]
        this.outputShape = this.inputShape;
    }

    GetNodeType = (): string => "CrossAttention";

    GetIsMerging = (): boolean => true;  // Signal that this is a merge node

    GetInfo = (): string => {
        return JSON.stringify({
            node: "CrossAttention",
            params: {
                n_heads: this.nHeads,
                dropout: this.dropout,
                quiet_softmax: this.quietSoftmax,
            },
        });
    };

    GetResources = (): Map<string, number> => {
        const resources = new Map<string, number>();
        const [seqLenQuery, dModel] = this.inputShape;

        // Context (key/value) shape must be computed from graph
        // Conservative estimate: assume seq_len_context ≈ seq_len_query
        const seqLenContext = seqLenQuery;

        // 4 projections: Q, K, V, Output
        const projectionParams = dModel * dModel + dModel;
        resources.set("trainable_params", projectionParams * 4);

        // Attention scores: [seq_len_query, seq_len_context]
        resources.set("intermediate_memory", seqLenQuery * seqLenContext * 4);
        return resources;
    };

    protected Mutate(options: Map<string, number>): void {
        const dModel = this.inputShape[1];
        const maxHeads = Math.min(Math.floor(dModel / 2), 16);

        if (Math.random() < 0.3) {
            this.nHeads = Math.max(1, Math.floor(Math.random() * maxHeads));
        }

        if (Math.random() < 0.3) {
            this.dropout = Math.random() * 0.5;
        }

        if (Math.random() < 0.2) {
            this.quietSoftmax = !this.quietSoftmax;
        }
    }

    protected GetExpectedInputDimensions(): number {
        return 2;
    }

    GetOutputDimensions(): number {
        return 2;
    }

    protected _CloneImpl(): BaseNode {
        const clone = new CrossAttentionNode(this.id, this.genomeId, this.inputShape);
        clone.nHeads = this.nHeads;
        clone.dropout = this.dropout;
        clone.quietSoftmax = this.quietSoftmax;
        clone.outputShape = this.outputShape;
        return clone;
    }

    // UI Properties
    SetNHeads(n: number): void {
        this.nHeads = Math.max(1, Math.min(n, 16));
    }

    GetNHeads(): number {
        return this.nHeads;
    }

    SetDropout(d: number): void {
        this.dropout = Math.max(0, Math.min(d, 1.0));
    }

    GetDropout(): number {
        return this.dropout;
    }
}
```

### 3.3 TransformerEncoderBlockNode

```typescript
import { BaseNode } from "../base_node";
import type { IShape } from "../../types";

export class TransformerEncoderBlockNode extends BaseNode {
    private nHeads: number = 4;
    private dFF: number = 512;
    private dropout: number = 0.1;
    private activation: string = "gelu";
    private normFirst: boolean = false;

    constructor(nodeId: string, genomeId: string, shape: IShape = [512, 64]) {
        super(nodeId, genomeId, shape);
        this.nodeType = "TransformerEncoderBlock";
    }

    protected CalculateOutputShape(): void {
        // Output shape = input shape
        this.outputShape = this.inputShape;
    }

    GetNodeType = (): string => "TransformerEncoderBlock";

    GetInfo = (): string => {
        return JSON.stringify({
            node: "TransformerEncoderBlock",
            params: {
                n_heads: this.nHeads,
                d_ff: this.dFF,
                dropout: this.dropout,
                activation: this.activation,
                norm_first: this.normFirst,
            },
        });
    };

    GetResources = (): Map<string, number> => {
        const resources = new Map<string, number>();
        const [_seqLen, dModel] = this.inputShape;

        // MultiHeadAttention: 4 projections
        const attentionParams = (dModel * dModel + dModel) * 4;

        // PositionWiseFeedForward: 2 dense layers
        // Linear(d_model, d_ff) + Linear(d_ff, d_model)
        const ffParams = dModel * this.dFF + this.dFF + this.dFF * dModel + dModel;

        // LayerNorm: 2 × (d_model + d_model) for weight + bias
        const normParams = (dModel + dModel) * 2;

        resources.set("trainable_params", attentionParams + ffParams + normParams);
        return resources;
    };

    protected Mutate(options: Map<string, number>): void {
        const dModel = this.inputShape[1];
        const maxHeads = Math.min(Math.floor(dModel / 2), 16);

        // N_heads: [2, 16] with constraint: dModel % nHeads == 0
        if (Math.random() < 0.3) {
            let nHeads = Math.max(2, Math.floor(Math.random() * maxHeads));
            while (dModel % nHeads !== 0) {
                nHeads = Math.max(2, nHeads - 1);
            }
            this.nHeads = nHeads;
        }

        // d_ff: [dModel, dModel * 8]
        if (Math.random() < 0.3) {
            this.dFF = Math.floor(
                dModel + Math.random() * (dModel * 7)
            );
        }

        // Dropout: [0.0, 0.5]
        if (Math.random() < 0.3) {
            this.dropout = Math.random() * 0.5;
        }

        // Activation swap
        if (Math.random() < 0.1) {
            this.activation = Math.random() < 0.5 ? "gelu" : "relu";
        }

        // norm_first flip
        if (Math.random() < 0.1) {
            this.normFirst = !this.normFirst;
        }
    }

    protected GetExpectedInputDimensions(): number {
        return 2;
    }

    GetOutputDimensions(): number {
        return 2;
    }

    protected _CloneImpl(): BaseNode {
        const clone = new TransformerEncoderBlockNode(this.id, this.genomeId, this.inputShape);
        clone.nHeads = this.nHeads;
        clone.dFF = this.dFF;
        clone.dropout = this.dropout;
        clone.activation = this.activation;
        clone.normFirst = this.normFirst;
        clone.outputShape = this.outputShape;
        return clone;
    }

    // UI Properties
    SetNHeads(n: number): void {
        const dModel = this.inputShape[1];
        this.nHeads = Math.max(1, Math.min(n, dModel));
    }

    GetNHeads(): number {
        return this.nHeads;
    }

    SetDFF(d: number): void {
        this.dFF = Math.max(this.inputShape[1], d);
    }

    GetDFF(): number {
        return this.dFF;
    }

    SetDropout(d: number): void {
        this.dropout = Math.max(0, Math.min(d, 1.0));
    }

    GetDropout(): number {
        return this.dropout;
    }
}
```

---

## 4. Backend Implementation (Rust/Burn)

### 4.1 Add DTO Variants (`src-tauri/src/dtos.rs`)

```rust
// Add to NodeDtoJSON enum:

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    // ... existing variants ...

    /// Multi-head Self-Attention Layer
    MultiHeadAttention {
        n_heads: u64,
        dropout: f64,
        quiet_softmax: bool,
    },

    /// Multi-head Cross-Attention Layer (merge node)
    CrossAttention {
        n_heads: u64,
        dropout: f64,
        quiet_softmax: bool,
    },

    /// Complete Transformer Encoder Block (MHA + FFNN + LayerNorms + Residuals)
    TransformerEncoderBlock {
        n_heads: u64,
        d_ff: u64,
        dropout: f64,
        activation: String,
        norm_first: bool,
    },
}
```

### 4.2 Module Storage in GraphModel (`src-tauri/src/entities.rs`)

Update the `GraphModel` struct:

```rust
#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    // ... existing fields ...
    
    /// Multi-head attention modules
    pub mha: Vec<MultiHeadAttention<B>>,
    
    /// Cross-attention modules
    pub cross_attn: Vec<CrossAttention<B>>,
    
    /// Transformer encoder block modules
    pub transformer_encoders: Vec<TransformerEncoderLayer<B>>,
    
    pub execution_plan: Ignored<Vec<Instruction>>,
}
```

### 4.3 Operation Enum (`src-tauri/src/entities.rs`)

Add operation variants:

```rust
#[derive(Clone, Debug)]
pub enum Operation {
    // ... existing variants ...
    
    /// Self-attention operation
    MultiHeadAttention { mha_idx: usize },
    
    /// Cross-attention operation (requires two inputs)
    CrossAttention { cross_attn_idx: usize },
    
    /// Transformer encoder block (complete layer)
    TransformerEncoderBlock { transformer_idx: usize },
}
```

### 4.4 Compilation Logic (`src-tauri/src/entities.rs::GraphModel::build()`)

```rust
impl<B: Backend> GraphModel<B> {
    fn build(genome: &GenomeDto, device: &B::Device) -> Self {
        let mut mha_modules = Vec::new();
        let mut cross_attn_modules = Vec::new();
        let mut transformer_encoder_modules = Vec::new();

        for node in &genome.nodes {
            match node.node_type {
                NodeDtoJSON::MultiHeadAttention {
                    n_heads,
                    dropout,
                    quiet_softmax,
                } => {
                    // Infer d_model from input shape
                    let d_model = node.output_shape[1] as usize; // [seq_len, d_model]

                    let mha = MultiHeadAttentionConfig::new(d_model, n_heads as usize)
                        .with_dropout(dropout)
                        .with_quiet_softmax(quiet_softmax)
                        .init(device);

                    mha_modules.push(mha);
                }

                NodeDtoJSON::CrossAttention {
                    n_heads,
                    dropout,
                    quiet_softmax,
                } => {
                    let d_model = node.output_shape[1] as usize;

                    let cross_attn = CrossAttentionConfig::new(d_model, n_heads as usize)
                        .with_dropout(dropout)
                        .init(device);

                    cross_attn_modules.push(cross_attn);
                }

                NodeDtoJSON::TransformerEncoderBlock {
                    n_heads,
                    d_ff,
                    dropout,
                    activation,
                    norm_first,
                } => {
                    let d_model = node.output_shape[1] as usize;

                    let encoder_config = TransformerEncoderConfig {
                        d_model,
                        d_ff: d_ff as usize,
                        n_heads: n_heads as usize,
                        n_layers: 1,  // Single block
                        dropout,
                        norm_first,
                        activation: ActivationConfig::from_string(&activation),
                        initializer: Initializer::default(),
                        layer_norm_eps: 1e-6,
                        quiet_softmax: false,
                    };

                    let encoder = encoder_config.init(device);
                    transformer_encoder_modules.push(encoder);
                }

                // ... other variants ...
            }
        }

        Self {
            mha: mha_modules,
            cross_attn: cross_attn_modules,
            transformer_encoders: transformer_encoder_modules,
            execution_plan: Ignored::new(execution_plan),
        }
    }
}
```

### 4.5 Forward Pass Implementation (`src-tauri/src/entities.rs::GraphModel::forward()`)

**Critical: Shape Reshaping**

Frontend sends 2D tensors `[seq_len, d_model]`. Burn MHA expects 3D: `[batch_size, seq_len, d_model]`.

**Solution**: Add batch dimension (batch_size=1) before, remove after.

```rust
impl<B: Backend> GraphModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut tensors: HashMap<usize, Tensor<B, 3>> = HashMap::new();

        // Initialize input: add batch dimension
        // [seq_len, d_model] → [1, seq_len, d_model]
        let [seq_len, d_model] = input.dims();
        let batched_input = input.unsqueeze::<3>(0);
        tensors.insert(input_node_id, batched_input);

        // Execute operations in topological order
        for instr in self.execution_plan.iter() {
            match instr {
                Operation::MultiHeadAttention { mha_idx } => {
                    let query = tensors.get(&instr.input_node_id)
                        .expect("Missing input tensor");

                    // MHA: self-attention
                    let mha_input = MhaInput::self_attn(query.clone());
                    let mha_output = self.mha[*mha_idx].forward(mha_input);
                    let context = mha_output.context;

                    tensors.insert(instr.node_id, context);
                }

                Operation::CrossAttention { cross_attn_idx } => {
                    // Cross-attention requires TWO inputs:
                    let query = tensors.get(&instr.input_nodes[0])
                        .expect("Missing query tensor");
                    let context = tensors.get(&instr.input_nodes[1])
                        .expect("Missing context tensor");

                    let output = self.cross_attn[*cross_attn_idx]
                        .forward(query.clone(), context.clone(), None);

                    tensors.insert(instr.node_id, output);
                }

                Operation::TransformerEncoderBlock { transformer_idx } => {
                    let input = tensors.get(&instr.input_node_id)
                        .expect("Missing input tensor");

                    let encoder_input = TransformerEncoderInput::new(input.clone());
                    let output = self.transformer_encoders[*transformer_idx]
                        .forward(encoder_input);

                    tensors.insert(instr.node_id, output);
                }

                // ... other operations ...
            }
        }

        // Extract final output and remove batch dimension
        let final_batched = tensors.get(&output_node_id).expect("Missing output");
        let [_batch, _seq_len, _d_model] = final_batched.dims();
        
        // [1, seq_len, d_model] → [seq_len, d_model]
        final_batched.squeeze::<2>(0)
    }
}
```

---

## 5. Shape Propagation Rules

### MultiHeadAttention

```
Input:      [seq_len, d_model]
Output:     [seq_len, d_model]

Rules:
- d_model must be divisible by n_heads
- Output sequence length = input sequence length
- d_model unchanged
```

### CrossAttention

```
Primary Input (query):  [seq_q, d_model]
Context Input (kv):     [seq_kv, d_model]
Output:                 [seq_q, d_model]

Rules:
- Both inputs must have same d_model
- Query and context can have different seq_lens
- Output length follows query length
```

### TransformerEncoderBlock

```
Input:      [seq_len, d_model]
Output:     [seq_len, d_model]

Constraints:
- d_model must be divisible by n_heads
- d_ff ≥ d_model (typically 4× d_model for efficiency)
```

### Compatibility Checks

Add to `CheckCompability()` function:

```typescript
// When connecting to MHA node:
if (targetNode instanceof MultiHeadAttentionNode) {
    const [_, dModel] = sourceNode.GetOutputDimensions();
    const nHeads = targetNode.GetNHeads();
    
    // Check: d_model % n_heads == 0
    if (dModel % nHeads !== 0) {
        return {
            compatible: false,
            reason: `d_model (${dModel}) must be divisible by n_heads (${nHeads})`
        };
    }
}

// When connecting to CrossAttention node (merge):
if (targetNode instanceof CrossAttentionNode) {
    // Check both inputs have same d_model
    const primaryDModel = sourceNode.GetOutputDimensions()[1];
    // Query input already validated
    // Need to ensure both paths have compatible d_models
}
```

---

## 6. Testing Strategy

### 6.1 TypeScript Unit Tests

```typescript
// multihead_attention_node.test.ts

import { MultiHeadAttentionNode } from "./multihead_attention_node";

describe("MultiHeadAttentionNode", () => {
    it("should calculate output shape correctly", () => {
        const node = new MultiHeadAttentionNode(
            "mha1",
            "genome1",
            [512, 64]  // [seq_len, d_model]
        );

        expect(node.GetOutputDimensions()).toEqual([512, 64]);
    });

    it("should mutate n_heads within valid range", () => {
        const node = new MultiHeadAttentionNode("mha1", "genome1", [512, 64]);
        for (let i = 0; i < 100; i++) {
            node.Mutate(new Map());
            expect(node.GetNHeads()).toBeGreaterThanOrEqual(1);
            expect(node.GetNHeads()).toBeLessThanOrEqual(32);
        }
    });

    it("should validate d_model divisibility by n_heads", () => {
        const node = new MultiHeadAttentionNode("mha1", "genome1", [512, 64]);
        const dModel = 64;

        for (let nHeads = 1; nHeads <= 16; nHeads++) {
            node.SetNHeads(nHeads);
            if (dModel % nHeads === 0) {
                expect(() => node.GetInfo()).not.toThrow();
            }
        }
    });

    it("should clone with all parameters preserved", () => {
        const original = new MultiHeadAttentionNode("mha1", "genome1", [512, 64]);
        original.SetNHeads(8);
        original.SetDropout(0.2);

        const clone = original._CloneImpl() as MultiHeadAttentionNode;

        expect(clone.GetNHeads()).toBe(8);
        expect(clone.GetDropout()).toBeCloseTo(0.2);
        expect(clone.GetOutputDimensions()).toEqual([512, 64]);
    });

    it("should calculate resource usage", () => {
        const node = new MultiHeadAttentionNode("mha1", "genome1", [512, 64]);
        const resources = node.GetResources();

        // 4 projections: d_model * d_model + d_model each
        // 64 * 64 + 64 = 4160 per projection = 16640 total
        expect(resources.get("trainable_params")).toBe(16640);

        // Attention scores: seq_len * seq_len
        // 512 * 512 = 262144 bytes
        expect(resources.get("intermediate_memory")).toBe(262144);
    });
});
```

### 6.2 Rust/Burn Backend Tests

```rust
// test_attention_nodes

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_forward_shape() {
        let device = TestBackend::device();
        
        // Create configuration
        let config = MultiHeadAttentionConfig::new(64, 4);
        let mha = config.init::<TestBackend>(&device);

        // Create input: [batch=1, seq_len=512, d_model=64]
        let batch_input = Tensor::random(
            [1, 512, 64],
            Distribution::Default,
            &device,
        );

        // Forward pass
        let mha_input = MhaInput::self_attn(batch_input.clone());
        let output = mha.forward(mha_input);

        // Verify output shape
        assert_eq!(output.context.shape().dims, [1, 512, 64]);
    }

    #[test]
    fn test_mha_divisibility_constraint() {
        let device = TestBackend::device();

        // Valid: 64 % 4 == 0
        let config = MultiHeadAttentionConfig::new(64, 4);
        let mha = config.init::<TestBackend>(&device);
        assert_eq!(mha.n_heads, 4);

        // Invalid: 64 % 5 != 0 (would panic in real usage)
        // This should be caught during genome compilation
    }

    #[test]
    fn test_transformer_encoder_block_forward() {
        let device = TestBackend::device();

        let config = TransformerEncoderConfig {
            d_model: 64,
            d_ff: 256,
            n_heads: 4,
            n_layers: 1,
            dropout: 0.1,
            norm_first: false,
            activation: ActivationConfig::Gelu,
            ..Default::default()
        };

        let encoder = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [1, 512, 64],
            Distribution::Default,
            &device,
        );

        let encoder_input = TransformerEncoderInput::new(input);
        let output = encoder.forward(encoder_input);

        assert_eq!(output.shape().dims, [1, 512, 64]);
    }
}
```

---

## 7. UI Integration

### 7.1 Parameter Editing Component

```typescript
// AttentionParametersPanel.tsx

interface AttentionParametersPanelProps {
    node: MultiHeadAttentionNode;
    onUpdate: (node: BaseNode) => void;
}

export function AttentionParametersPanel({
    node,
    onUpdate,
}: AttentionParametersPanelProps) {
    const [nHeads, setNHeads] = useState(node.GetNHeads());
    const [dropout, setDropout] = useState(node.GetDropout());
    const [quietSoftmax, setQuietSoftmax] = useState(node.GetQuietSoftmax());

    const handleNHeadsChange = (value: number) => {
        setNHeads(value);
        node.SetNHeads(value);
        onUpdate(node);
    };

    const handleDropoutChange = (value: number) => {
        setDropout(value);
        node.SetDropout(value);
        onUpdate(node);
    };

    const handleQuietSoftmaxChange = (value: boolean) => {
        setQuietSoftmax(value);
        node.SetQuietSoftmax(value);
        onUpdate(node);
    };

    return (
        <div className={styles.panel}>
            <label>
                Number of Heads
                <input
                    type="number"
                    min="1"
                    max="16"
                    value={nHeads}
                    onChange={(e) => handleNHeadsChange(Number(e.target.value))}
                />
            </label>

            <label>
                Dropout
                <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.05"
                    value={dropout}
                    onChange={(e) => handleDropoutChange(Number(e.target.value))}
                />
                <span>{(dropout * 100).toFixed(0)}%</span>
            </label>

            <label>
                <input
                    type="checkbox"
                    checked={quietSoftmax}
                    onChange={(e) => handleQuietSoftmaxChange(e.target.checked)}
                />
                Quiet Softmax (Gemma-style)
            </label>
        </div>
    );
}
```

### 7.2 Node Factory Registration

```typescript
// In node creation factory
export function createNodeByType(
    type: string,
    nodeId: string,
    genomeId: string,
    inputShape: IShape
): BaseNode {
    switch (type) {
        case "MultiHeadAttention":
            return new MultiHeadAttentionNode(nodeId, genomeId, inputShape);
        
        case "CrossAttention":
            return new CrossAttentionNode(nodeId, genomeId, inputShape);
        
        case "TransformerEncoderBlock":
            return new TransformerEncoderBlockNode(nodeId, genomeId, inputShape);

        // ... other types ...
    }
}
```

---

## 8. Advanced Patterns

### 8.1 Causal Masking for Autoregressive Models

For language modeling / sequence generation, apply causal mask:

```rust
// In forward pass, before MHA:
let [batch, seq_len] = input.dims();

// Create lower-triangular bool mask (causal)
let mask_attn = Tensor::<B, 3, Bool>::zeros(
    [batch, seq_len, seq_len],
    &device,
);
// Set upper triangle to true (masked positions)
for i in 0..seq_len {
    for j in (i+1)..seq_len {
        mask_attn[batch][i][j] = true;
    }
}

let input_mha = MhaInput::self_attn(input)
    .mask_attn(mask_attn);

let output = self.mha.forward(input_mha);
```

**Frontend Support**: Add parameter `is_causal: bool` to MHA node.

### 8.2 Stacking Multiple Transformer Layers

Instead of single `TransformerEncoderBlockNode`, create architecture that chains multiple:

```
Input → MHA₁ → Dense → MHA₂ → Dense → ... → Output
```

Or use wrapped `TransformerEncoderNode` that creates full stack:

```rust
// In dtos for 12-layer transformer stack:
TransformerStack {
    d_model: 768,
    n_layers: 12,
    n_heads: 12,
    d_ff: 3072,
    dropout: 0.1,
}
```

### 8.3 Positional Encoding

**Note**: Standard Transformer implementations require positional encodings. For the neural-evo app:

**Option 1**: Include PE in separate `PositionalEncodingNode` (wrapper around Burn's RotaryEmbedding or similar)

```typescript
export class PositionalEncodingNode extends BaseNode {
    GetNodeType = (): string => "PositionalEncoding";
    
    protected CalculateOutputShape(): void {
        this.outputShape = this.inputShape;  // PE doesn't change shape
    }
}
```

**Option 2**: Include PE internally in TransformerEncoderBlockNode (simpler but less flexible)

**Option 3**: Let user apply Dense transform to learnable positional embeddings (most evolutionary-friendly)

For MVP, **Option 3** recommended: user can add learnable embeddings via Dense layers.

### 8.4 Cross-Attention Graph Topology

Standard graphs are DAGs with single input per node. Cross-attention nodes break this:

**Graph Representation Strategy**:

```
Encoder branch:
Input → Dense → Conv1D → MHA → Output₁

Decoder branch:
Input → LSTM → CrossAttention (consumes Output₁)

In genome:
edges: [
    { source: Input, target: Dense },
    { source: Dense, target: Conv1D },
    { source: Conv1D, target: MHA },
    { source: MHA, target: Output₁, genomeId: encoder_id },
    
    { source: Input, target: LSTM },
    { source: LSTM, target: CrossAttention, input_index: 0 },  // query
    { source: Output₁, target: CrossAttention, input_index: 1 }, // context
    { source: CrossAttention, target: Output }
]
```

**Implementation**: Extend graph edge structure to support `input_index` field for merge nodes.

---

## Implementation Checklist

### Phase 1: MultiHeadAttentionNode
- [ ] Create `src/entities/canvas-genome/model/nodes/attention/multihead_attention.ts`
- [ ] Implement all 9 abstract methods
- [ ] Create unit tests
- [ ] Register in node factory
- [ ] Add DTO variant (NodeDtoJSON::MultiHeadAttention)
- [ ] Add to GraphModel module storage
- [ ] Implement build() compilation logic
- [ ] Implement forward() operation
- [ ] Create UI parameter panel

### Phase 2: TransformerEncoderBlockNode
- [ ] Create `src/entities/canvas-genome/model/nodes/attention/transformer_encoder_block.ts`
- [ ] Implement 9 abstract methods
- [ ] Create unit tests
- [ ] Register in factory
- [ ] Add DTO + build() + forward()
- [ ] Create UI panel

### Phase 3: CrossAttentionNode (Advanced)
- [ ] Create `src/entities/canvas-genome/model/nodes/attention/cross_attention.ts`
- [ ] Implement merge logic
- [ ] Extend graph topology to support dual inputs
- [ ] Add DTO + build() + forward()
- [ ] Create UI panel
- [ ] Test encoder-decoder patterns

### Phase 4: Integration & Testing
- [ ] Compatibility checks in `CheckCompability()`
- [ ] Shape propagation rules
- [ ] End-to-end training examples
- [ ] Documentation updates

---

## Known Limitations & Workarounds

| Issue | Impact | Workaround |
|-------|--------|-----------|
| No built-in positional encoding node | Transformers may underperform without PE | Add learnable Dense embeddings in genome design |
| Cross-attention requires dual inputs | Graph topology complexity | Use merge node pattern; simplest for MVP |
| No relative position bias (ALiBi) | Limited bias variety | Can be added as future Enhancement |
| Causal masking requires frontend support | Autoregressive models harder | Add `is_causal` boolean param to MHA |
| Memory usage scales as O(seq_len²) | Problematic for long sequences | Document seq_len < 2048 recommendation |

---

## References

- [Burn Attention Module API](https://burn-rs.github.io/docs/burn/nn/modules/attention/index.html)
- [Transformer Architecture Paper](https://arxiv.org/abs/1706.03762)
- [Burn Transformer Examples](https://github.com/tracel-ai/burn/tree/main/examples/text-generation)
- [Existing LSTM/GRU Implementation Guide](./LSTM_GRU_IMPLEMENTATION_GUIDE.md)
