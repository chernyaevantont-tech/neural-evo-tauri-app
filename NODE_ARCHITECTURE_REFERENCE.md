# Neural Network Node Architecture — Visual Reference & Quick Guide

## Class Hierarchy Diagram

```
BaseNode (Abstract)
├── Layers (Output dimensions fixed, single input)
│   ├── InputNode
│   ├── DenseNode
│   ├── Conv2DNode
│   ├── PoolingNode (max/avg)
│   ├── FlattenNode
│   ├── OutputNode
│   └── BatchNormNode
├── Regularization (Pass-through, single input)
│   ├── DropoutNode
│   ├── Dropout2DNode
│   ├── LayerNormNode
│   └── GaussianNoiseNode
└── Merge Layers (Multi-input, GetIsMerging()=true)
    ├── AddNode
    └── Concat2DNode
```

---

## Node Type Reference Table

| Node Type | Input Dims | Output Dims | Parameters | Burn Module | Use Case |
|-----------|-----------|-----------|-----------|-----------|----------|
| **Input** | - | any | output_shape | None | Entry point, defines image/feature dimensions |
| **Dense** | 1 | 1 | units, activation, use_bias | Linear | Fully connected classification/regression layer |
| **Conv2D** | 3 | 3 | filters, kernel, stride, padding, dilation, activation | Conv2d | Feature extraction from images |
| **Pooling** | 3 | 3 | pool_type, kernel, stride, padding | MaxPool2d/AvgPool2d | Dimensionality reduction, translation invariance |
| **Flatten** | 3 | 1 | - | None (reshape) | Bridge Conv→Dense |
| **Output** | any | - | input_shape | None | Terminal node, defines target shape |
| **Add** | any | any | - | None | Residual connections, multi-branch fusion |
| **Concat** | 3 | 3 | - | None | Feature concatenation on channel axis |
| **BatchNorm** | any | any | epsilon, momentum | BatchNorm | Normalization, training stabilization |
| **Dropout** | 1 | 1 | prob | Dropout | Regularization (dense) |
| **Dropout2D** | 3 | 3 | prob | Dropout | Regularization (spatial) |
| **LayerNorm** | 1 | 1 | epsilon | LayerNorm | Normalization for dense layers |
| **GaussianNoise** | any | any | std_dev | None | Augmentation during training |

---

## Shape Propagation Examples

### Example 1: Simple Sequential Network

```
Input [28, 28, 3]
    ↓ (PropagateShapeUpdate)
Conv2D (filters=32, k=3, s=1, p=1)
    inputShape = [28, 28, 3]
    outputShape = [28, 28, 32]  ← propagates to next
    ↓
Pooling (max, k=2, s=2, padding=0)
    inputShape = [28, 28, 32]
    outputShape = [14, 14, 32]
    ↓
Flatten
    inputShape = [14, 14, 32]
    outputShape = [6272]  ← 14*14*32
    ↓
Dense (units=128, activation=relu)
    inputShape = [6272]
    outputShape = [128]
    ↓
Output
    inputShape = [128]
```

### Example 2: Branch & Merge (ResNet-like)

```
    ┌─────────────────────────────────────────┐
    │                                         │
Input [H, W, C]                        Identity branch
    │                                         │
    ↓                                         │
Conv2D(64) → outputShape [H, W, 64]          │
    ↓                                    (no change)
BatchNorm → outputShape [H, W, 64]           │
    ↓                                         │
Conv2D(64) → outputShape [H, W, 64] ←────────┤
    ↓                                         │
    └─────────→ AddNode ←──────────────────────┘
        outputShape [H, W, 64]

**Requirement**: All paths to AddNode must have identical shapes!
```

**Compatibility Check at AddNode**:
```typescript
// If Conv2D output ≠ Identity branch → ERROR
// Both must produce [H, W, 64] exactly
```

---

## Parameter Mutation Space

### DenseNode
```
units:      2^4 to 2^12  = [16, ..., 4096]
activation: ["relu", "leaky_relu", "softmax"]
use_bias:   boolean toggle
```

### Conv2DNode
```
filters:    4 × [4 to 16]  = [16, 20, 24, ..., 64]
kernel:     odd sizes [1, 3, 5, 7] (via 1 + 2*RandomizeInteger(0,3))
stride:     [1, 2]
padding:    [1, 2]
dilation:   [1, 2, 4, 8]
activation: ["relu", "leaky_relu", "sigmoid", "linear"]
use_bias:   boolean toggle
```

### PoolingNode
```
pool_type:  ["max", "avg"]
kernel:     fixed at construction
stride:     fixed at construction
padding:    fixed at construction
```

---

## Serialization Format

### JSON Structure (Frontend → Backend)

```json
{
  "node": "Dense",
  "params": {
    "units": 256,
    "activation": "relu",
    "use_bias": true
  }
}
```

### String Format (Full Genome)

```
{"node":"Input","params":{"output_shape":[28,28,3]}}
{"node":"Conv2D","params":{"filters":32,"kernel_size":{"h":3,"w":3},"stride":1,"padding":1,"dilation":1,"use_bias":true,"activation":"relu"}}
{"node":"Dense","params":{"units":128,"activation":"relu","use_bias":true}}
{"node":"Output","params":{"input_shape":[128]}}
CONNECTIONS
0 1
1 2
2 3
```

**Key**: Index-based edges, topological sorting guaranteed by BFS during serialization.

---

## BaseNode Interface Contract

Every node **must** implement these methods:

```typescript
// 1. TYPE DISCRIMINATION
public GetNodeType(): string                    // "Dense", "Conv2D", etc.

// 2. STRUCTURAL VALIDATION
public GetExpectedInputDimensions(): number | "any"
public GetOutputDimensions(): number | "any"
public GetIsMerging(): boolean                  // true = multi-input allowed

// 3. GRAPH OPERATIONS
public CanAcceptConnectionFrom(node, check?): boolean
    → Can this node receive input from `node`?

// 4. SHAPE PROPAGATION
protected CalculateOutputShape(): void          // Compute outputShape from inputShape
    → Called by PropagateShapeUpdate()

public PropagateShapeUpdate(visited): void      // Recursive cascade downstream
    → Updates this and all next nodes

// 5. SERIALIZATION
public GetInfo(): string                        // JSON serialize: {"node":"...", "params":{...}}

// 6. RESOURCE ESTIMATION
public GetResources(dtype): ResourceCriteria    // flash, ram, macs

// 7. EVOLUTIONARY MUTATION
protected Mutate(mutation_options): void        // Randomize parameters

// 8. CLONING (Genetic Operators)
public Clone(): BaseNode                        // Deep copy with new UUID
protected _CloneImpl(): BaseNode                 // Subclass implementation
```

---

## Polymorphism: Method Override Patterns

### Pattern 1: Shape Calculation Override

| Node | Override Details |
|------|------------------|
| **DenseNode** | `outputShape = [units]` |
| **Conv2DNode** | `outputShape = [out_h, out_w, filters]` using convolution formula |
| **FlattenNode** | `outputShape = [H × W × C]` flattening |
| **PoolingNode** | `outputShape = [out_h, out_w, channels]` preserving channels |
| **AddNode** | `outputShape = inputShape` (pass-through) |
| **Concat2DNode** | `outputShape = [H, W, Σ channels]` summing channel dimension |
| **BatchNormNode** | `outputShape = inputShape` (pass-through) |

### Pattern 2: Input Validation Override

**BaseNode** (default):
```typescript
public CanAcceptConnectionFrom(node: BaseNode): boolean {
    if (!this.GetIsMerging() && this.previous.length >= 1) {
        return false;  // Single-input only
    }
    // Check dimension compatibility...
    return sourceDims === targetExpected || targetExpected === "any";
}
```

**AddNode** (strict shape matching):
```typescript
public CanAcceptConnectionFrom(node: BaseNode): boolean {
    if (!super.CanAcceptConnectionFrom(node)) return false;
    
    const targetShape = this.GetInputShape();
    const incShape = node.GetOutputShape();
    
    // All inputs must match exactly (element-wise addition)
    if (targetShape.length !== incShape.length || 
        !targetShape.every((v, i) => v === incShape[i])) {
        return false;
    }
    return true;
}
```

**Concat2DNode** (spatial consistency):
```typescript
public CanAcceptConnectionFrom(node: BaseNode): boolean {
    if (!super.CanAcceptConnectionFrom(node)) return false;
    
    const targetShape = this.GetInputShape();
    const incShape = node.GetOutputShape();
    
    // H and W must match; channels will be concatenated
    if (targetShape.length === 3 && incShape.length === 3) {
        if (targetShape[0] !== incShape[0] ||  // H must match
            targetShape[1] !== incShape[1]) {  // W must match
            return false;
        }
    }
    return true;
}
```

### Pattern 3: Mutation Space Override

| Node | Mutation Strategy |
|------|------------------|
| **DenseNode** | Units (power-of-2), activation, bias toggle |
| **Conv2DNode** | Filters, kernel size, stride, padding, dilation, activation |
| **PoolingNode** | Pool type (max/avg) only |
| **FlattenNode** | No mutations (pure reshape) |
| **BatchNormNode** | No mutations (parameters set at construction) |
| **InputNode** | No mutations (fixed by problem definition) |

---

## Burn Backend Translation Table

### NodeDtoJSON → Burn Module Mapping

| Frontend Node | Burn Module | Config Call |
|--------------|-------------|------------|
| Dense(units, activation, bias) | Linear | `LinearConfig::new(input_dim, units).with_bias(bias).init(device)` |
| Conv2D(filters, kernel, stride, padding, dilation, bias, activation) | Conv2d | `Conv2dConfig::new([in_ch, filters], [k_h, k_w]).with_stride([...]).with_padding(...).with_dilation(...).with_bias(bias).init(device)` |
| Pooling(pool="max", kernel, stride, padding) | MaxPool2d | `MaxPool2dConfig::new([k_h, k_w]).with_strides([...]).with_padding(...).init()` |
| Pooling(pool="avg", kernel, stride, padding) | AvgPool2d | `AvgPool2dConfig::new([k_h, k_w]).with_strides([...]).with_padding(...).init()` |
| Dropout(prob) | Dropout | `DropoutConfig::new(prob).init()` |
| BatchNorm(epsilon, momentum) | BatchNorm | `BatchNormConfig::new(channels).with_epsilon(epsilon).with_momentum(momentum).init(device)` |
| LayerNorm(epsilon) | LayerNorm | `LayerNormConfig::new(features).with_epsilon(epsilon).init(device)` |
| Flatten, Add, Concat | None | Implemented in forward() as tensor operations |

### Operation Enum (Internal Representation)

```rust
pub enum Operation {
    Input(usize),                                      // Input index
    Dense { dense_idx: usize, activation: String },   // Ref Linear module + activation name
    Conv2D { conv2d_idx: usize, activation: String }, // Ref Conv2d module + activation name
    MaxPool { pool_idx: usize },                       // Ref MaxPool2d module
    AvgPool { pool_idx: usize },                       // Ref AvgPool2d module
    Flatten,                                           // Pure tensor op
    Add,                                               // Pure tensor op
    Concat,                                            // Pure tensor op
    Dropout { dropout_idx: usize },                    // Ref Dropout module
    BatchNorm { batch_norm_idx: usize },               // Ref BatchNorm module
    LayerNorm { layer_norm_idx: usize },               // Ref LayerNorm module
    Dropout2D { dropout_2d_idx: usize },               // Ref Dropout module (spatial)
    GaussianNoise { std_dev: f64 },                    // Carry parameter directly
    Output(usize),                                     // Output index
}
```

---

## Activation Function Handling

### Frontend (DenseNode)
```typescript
activation: "relu" | "leaky_relu" | "softmax"
```

### Backend (Operation)
```rust
Operation::Dense { activation: "relu" } → forward():
    out = relu(linear.forward(x))

Operation::Dense { activation: "ldaky_relu" } → forward():
    out = leaky_relu(linear.forward(x), 0.01)

Operation::Dense { activation: "softmax" } → forward():
    out = softmax(linear.forward(x), 1)

Operation::Dense { activation: "linear" } → forward():
    out = linear.forward(x)  // No activation
```

### Special Case: Output Dense Auto-Conversion
```rust
// In compilation phase:
if is_output_layer && activation == "softmax" && units > 1 {
    // Auto-convert softmax → linear
    // Reason: CrossEntropyLoss expects logits, not probabilities
    final_activation = "linear"
}
```

---

## Dimension Constraints (Input/Output)

### By Layer Type

```
Input:        handles any → can be 1D [batch] or 3D [H,W,C]
              GetOutputDimensions() = this.outputShape.length

Dense:        1D input → 1D output
              GetExpectedInputDimensions() = 1
              GetOutputDimensions() = 1

Conv2D:       3D input → 3D output
              GetExpectedInputDimensions() = 3
              GetOutputDimensions() = 3

Pooling:      3D input → 3D output
              GetExpectedInputDimensions() = 3
              GetOutputDimensions() = 3

Flatten:      3D input → 1D output
              GetExpectedInputDimensions() = 3
              GetOutputDimensions() = 1

Output:       flexible input (stored in node)
              GetExpectedInputDimensions() = "any"

Add:          flexible, but ALL inputs must match
              GetExpectedInputDimensions() = "any"
              (enforced by CanAcceptConnectionFrom override)

Concat:       3D input → 3D output (H,W must match)
              GetExpectedInputDimensions() = 3
              GetOutputDimensions() = 3
```

### Valid Connection Examples

✓ Input[3D] → Conv2D → Conv2D → Flatten → Dense → Dense → Output
✓ Input[3D] → Conv2D → Flatten → Dense → Output
✓ Input[1D] → Dense → Dense → Output
✓ Conv2D[3D] → AddNode ← Conv2D[3D]  (both produce [H,W,C])

✗ Input[3D] → **Dense** (Dense expects 1D)
✗ Conv2D → **Add** ← Dense (Conv2D→[3D], Dense→[1D], shapes don't match)
✗ Conv2D[32] → **Add** ← Conv2D[64] (different channels)

---

## Frontend ↔ Backend Format Conversions

### Tensor Shape Convention Mismatch

**Frontend** (canvas representation, HWC order):
```
Conv2D input: [Height, Width, Channels]   = [28, 28, 3]
Dense input:  [Features]                   = [784]
```

**Burn** (CHW order for spatial, standard for dense):
```
Conv2D input: [Channels, Height, Width]   = [3, 28, 28]
Dense input:  [Batch, Features]           = [batch, 784]
```

**Conversion** (in Rust Input node processing):
```rust
NodeDtoJSON::Input { output_shape } => {
    let shape = output_shape;  // [28, 28, 3]
    
    // If 3D (spatial), rotate to CHW
    let internal_shape = if shape.len() == 3 {
        vec![shape[2], shape[0], shape[1]]  // [3, 28, 28]
    } else {
        shape  // 1D stays as-is
    };
}
```

---

## Testing & Validation Checklist

### Shape Propagation Tests
- [ ] Input → Conv2D: shape[0,1] change, channels become filter count
- [ ] Conv2D → Pooling: spatial dimensions shrink, channels preserved
- [ ] Conv2D → Flatten: 3D → 1D correctly computed
- [ ] Flatten → Dense: flattened size matches input dimension
- [ ] Dense → Dense: output units become next input dimension

### Merge Node Tests
- [ ] AddNode: both inputs same shape → success
- [ ] AddNode: inputs different shapes → error
- [ ] Concat2DNode: both inputs [H,W,C1] and [H,W,C2] → output [H,W,C1+C2]
- [ ] Concat2DNode: different H or W → error

### Serialization Tests
- [ ] DenseNode → JSON → DenseNode: parameters preserved
- [ ] Conv2DNode → JSON → Conv2DNode: all parameters preserved
- [ ] Graph structure: edges serialized correctly
- [ ] Deserialization: topology reconstructed, shape propagation recomputed

### Backend Tests
- [ ] Dense JSON → Linear module created with correct dims
- [ ] Conv2D JSON → Conv2d module created with correct config
- [ ] Forward pass: output shapes match expected
- [ ] Activation functions applied correctly

---

## Common Bugs & Debugging

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| "Cannot connect nodes" | Dimension mismatch | `CanAcceptConnectionFrom()` validation |
| Shape shows NaN or 0 | Convolution formula error or padding too large | `CalculateOutputShape()` logic |
| AddNode connection fails | Shape mismatch between input branches | Both paths to AddNode must produce identical shapes |
| Rust compilation fails | NodeDtoJSON deserialization mismatch | JSON "node" field must match enum variant exactly |
| Forward pass crashes | Tensor dimension mismatch at runtime | Check shape cache consistency in topological sort |
| Mutation produces invalid network | Mutation exceeds bounds | Check RandomizeInteger ranges, validate post-mutation |

---

## Quick Reference: Adding a New Node Type

1. **Create TypeScript class** extending BaseNode in `src/entities/canvas-genome/model/nodes/`:
   ```typescript
   export class MyNode extends BaseNode {
       constructor(...params) { super(); }
       protected CalculateOutputShape(): void { /* formula */ }
       abstract GetNodeType(): string { return "MyNode"; }
       // ... implement all abstract methods
   }
   ```

2. **Add Rust NodeDtoJSON variant** in `src-tauri/src/dtos.rs`:
   ```rust
   #[derive(Deserialize)]
   pub enum NodeDtoJSON {
       MyNode { param1: u64, param2: String },
       // ...
   }
   ```

3. **Add compilation logic** in `src-tauri/src/entities.rs`:
   ```rust
   NodeDtoJSON::MyNode { param1, param2 } => {
       // Create Burn module
       // Return (Operation::MyNode { ... }, output_shape)
   }
   ```

4. **Add deserialization** in `src/entities/canvas-genome/lib/deserializeGenome.ts`:
   ```typescript
   case "MyNode":
       nodes.push(new MyNode(obj.params.param1, obj.params.param2));
       break;
   ```

5. **Test**:
   - Shape propagation from input
   - Serialization/deserialization round-trip
   - Rust compilation and forward pass

---

This reference guide pairs with `TECHNICAL_NODE_ARCHITECTURE.md` for complete understanding!
