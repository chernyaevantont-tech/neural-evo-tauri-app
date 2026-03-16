# Neural Network Node Architecture — Technical Deep Dive

## Table of Contents
1. [Frontend Node Architecture (BaseNode & Implementations)](#frontend-architecture)
2. [Backend Burn Translation (src-tauri)](#backend-translation)
3. [Polymorphism & Type Discrimination](#polymorphism)
4. [Serialization/Deserialization Pipeline](#serialization)
5. [Complete Examples: Dense, Conv2D, Flatten, AddNode](#examples)

---

## 1. Frontend Node Architecture

### 1.1 BaseNode Abstract Class

The **BaseNode** is the foundational abstract class all node types inherit from. It defines the interface for graph-based neural network construction.

**File**: [src/entities/canvas-genome/model/nodes/base_node.ts](src/entities/canvas-genome/model/nodes/base_node.ts)

#### Abstract Properties & Methods

```typescript
export abstract class BaseNode {
    // Graph connectivity
    public previous: BaseNode[] = []      // Incoming nodes (predecessors)
    public next: BaseNode[] = []          // Outgoing nodes (successors)
    
    // Shape tracking
    protected inputShape: number[] = []   // Input tensor shape
    protected outputShape: number[] = []  // Output tensor shape
    
    // Identification
    public id: string = v4()              // Unique UUID
    public innovationId: number = getNextInnovationId()  // Evolutionary tracking
    
    // Abstract methods (must be implemented by subclasses)
    protected abstract CalculateOutputShape(): void
    abstract GetInfo(): string
    abstract GetResources(dtype: number): ResourceCriteria
    protected abstract Mutate(mutation_options: Map<string, number>): void
    abstract GetExpectedInputDimensions(): number | "any"
    abstract GetOutputDimensions(): number | "any"
    abstract GetNodeType(): string
    protected abstract _CloneImpl(): BaseNode
    abstract GetIsMerging(): boolean
}
```

#### Key Methods

| Method | Purpose |
|--------|---------|
| `GetOutputShape()` | Returns calculated output tensor shape |
| `GetInputShape()` | Returns expected input tensor shape |
| `PropagateShapeUpdate(visited: Set<string>)` | **Recursive shape propagation**: Updates this node and all downstream nodes when input shape changes |
| `AddNext(node: BaseNode)` | Creates directed edge `this → node` and triggers shape propagation |
| `CheckCompability(node: BaseNode)` | Validates if this node can connect to `node` (acyclicity + dimension matching) |
| `CanAcceptConnectionFrom(node, isDisconnectedCheck?)` | Structural validation; merge nodes override for multi-input support |
| `Clone()` | Deep copy with fresh UUID; preserves parameters |
| `isAcyclic()` | DFS-based cycle detection |

#### Shape Propagation (Core Logic)

```typescript
public PropagateShapeUpdate(visited: Set<string> = new Set()): void {
    if (visited.has(this.id)) return;
    visited.add(this.id);

    // 1. Recalculate THIS node's output shape
    this.CalculateOutputShape();
    
    // 2. For non-merge nodes, set input shape of next nodes
    this.next.forEach(n => {
        if (!n.GetIsMerging()) {
            n.SetInputShape(this.outputShape);
        }
        // 3. Recursively propagate downstream
        n.PropagateShapeUpdate(visited);
    });
}
```

**Flow Example**: InputNode [28, 28, 3] → Conv2D → outputShape [26, 26, 32] → next node's inputShape becomes [26, 26, 32].

---

### 1.2 Node Type Discriminator

All nodes use the **`GetNodeType()`** method to identify themselves:

```typescript
public GetNodeType(): string  // Returns: "Dense" | "Conv2D" | "Input" | "Add" | etc.
```

This is used during **serialization/deserialization** to reconstruct the correct class instance.

**Type List** (13 node types):
- **Layers**: `Input`, `Dense`, `Conv2D`, `Pooling`, `Flatten`, `Output`
- **Merge**: `Add`, `Concat`
- **Regularization**: `BatchNorm`, `LayerNorm`, `Dropout`, `Dropout2D`, `GaussianNoise`

---

### 1.3 Resource Tracking

Each node can report computational costs:

```typescript
export type ResourceCriteria = {
    flash: number  // Parameter storage (bytes)
    ram: number    // Activation memory (bytes)
    macs: number   // Multiply-accumulate operations
}
```

**Example from DenseNode**:
```typescript
GetResources(dtype: number): ResourceCriteria {
    const flash = this.outputShape[0] * (this.inputShape[0] + (this.useBias ? 1 : 0)) * dtype
    const ram = this.inputShape[0] * this.outputShape[0] * dtype
    const macs = this.inputShape[0] * this.outputShape[0]
    return { flash, ram, macs }
}
```

Used for **parsimony pressure** fitness evaluation and resource-constrained optimization.

---

## 2. Node Type Implementations

### 2.1 DenseNode (Fully Connected Layer)

**File**: [src/entities/canvas-genome/model/nodes/layers/dense_node.ts](src/entities/canvas-genome/model/nodes/layers/dense_node.ts)

```typescript
export class DenseNode extends BaseNode {
    private units: number                              // Output dimensionality
    private activation: ActivationFunction             // "relu" | "leaky_relu" | "softmax"
    private useBias: Boolean
    private static activationFunctions = ["relu", "leaky_relu", "softmax"]

    constructor(units: number, activation: ActivationFunction, useBias: Boolean) {
        super()
        this.units = units
        this.activation = activation
        this.useBias = useBias
        this.inputShape = new Array<number>(1)  // Expects 1D input
    }

    protected CalculateOutputShape(): void {
        this.outputShape = [this.units]  // Always 1D output
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 1  // Must have 1D input
    }

    public GetOutputDimensions(): number | "any" {
        return 1  // Always outputs 1D
    }

    GetInfo(): string {
        return JSON.stringify({
            node: "Dense",
            params: {
                units: this.units,
                activation: this.activation,
                use_bias: this.useBias
            }
        })
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        // Mutation: randomly change units to power-of-2 between 2^4=16 and 2^12=4096
        if (Math.random() <= (mutation_options.get("dense_units") || -1)) {
            this.units = Math.pow(2, RandomizeInteger(4, 12))
        }
        
        // Mutation: switch activation function
        if (Math.random() <= (mutation_options.get("dense_activation") || -1)) {
            this.activation = DenseNode.activationFunctions[RandomizeInteger(0, 2)]
        }
        
        // Mutation: toggle bias
        if (Math.random() <= (mutation_options.get("dense_use_bias") || -1)) {
            this.useBias = !this.useBias
        }
        
        this.CalculateOutputShape()
    }

    public GetNodeType = (): string => "Dense"

    protected _CloneImpl = (): BaseNode => new DenseNode(this.units, this.activation, this.useBias)

    public GetIsMerging = (): boolean => false
}
```

**Constraints**:
- Input must be 1D (`[batch_size, features]` in Burn terms)
- Output is 1D → can connect to other Dense layers or Output directly
- **Parsimony guard**: Total parameters capped at **50M** to prevent memory overflow

---

### 2.2 Conv2DNode (2D Convolutional Layer)

**File**: [src/entities/canvas-genome/model/nodes/layers/conv_node.ts](src/entities/canvas-genome/model/nodes/layers/conv_node.ts)

```typescript
export class Conv2DNode extends BaseNode {
    private static dilationOptions = [1, 2, 4, 8]
    private filters: number                   // Output channels
    private kernelSize: KernelSize            // {h, w}
    private stride: number
    private padding: number
    private dilation: number
    private useBias: boolean
    private activation: string

    constructor(
        filters: number,
        kernelSize: KernelSize,
        stride: number,
        padding: number,
        dilation: number,
        useBias: boolean,
        activation: string = 'relu'
    ) {
        super()
        this.filters = filters
        this.kernelSize = kernelSize
        this.stride = stride
        this.padding = padding
        this.dilation = dilation
        this.useBias = useBias
        this.activation = activation
        this.inputShape = new Array<number>(3)  // [H, W, C] format
    }

    protected CalculateOutputShape(): void {
        // Standard convolution output shape formula
        const hOut = Math.floor(
            (this.inputShape[0] + 2*this.padding - this.dilation*(this.kernelSize.h - 1) - 1) 
            / this.stride + 1
        )
        const wOut = Math.floor(
            (this.inputShape[1] + 2*this.padding - this.dilation*(this.kernelSize.w - 1) - 1) 
            / this.stride + 1
        )
        this.outputShape = [hOut, wOut, this.filters]
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 3  // Must have 3D input: [H, W, C]
    }

    public GetOutputDimensions(): number | "any" {
        return 3  // Always outputs 3D
    }

    GetInfo(): string {
        return JSON.stringify({
            node: "Conv2D",
            params: {
                filters: this.filters,
                kernel_size: this.kernelSize,
                stride: this.stride,
                padding: this.padding,
                dilation: this.dilation,
                use_bias: this.useBias,
                activation: this.activation
            }
        })
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("conv2d_filters") || -1)) {
            this.filters = 4 * RandomizeInteger(4, 16)  // [16, 64] in multiples of 4
        }
        if (Math.random() <= (mutation_options.get("conv2d_kernel_size") || -1)) {
            const kernelSize = 1 + 2 * RandomizeInteger(0, 3)  // [1, 3, 5, 7]
            this.kernelSize = { h: kernelSize, w: kernelSize }
        }
        if (Math.random() <= (mutation_options.get("conv2d_dilation") || -1)) {
            this.dilation = Conv2DNode.dilationOptions[RandomizeInteger(0, 3)]
        }
        // ... other mutations
        this.CalculateOutputShape()
    }

    public GetNodeType = (): string => "Conv2D"
    protected _CloneImpl = (): BaseNode => new Conv2DNode(...)
    public GetIsMerging = (): boolean => false
}
```

**Computational Cost**:
```
flash = filters × (kernel_h × kernel_w × input_channels + bias) × dtype
ram = (input_h × input_w × input_c + output_h × output_w × filters) × dtype
macs = output_h × output_w × filters × kernel_h × kernel_w × input_c
```

---

### 2.3 AddNode (Merge: Element-wise Addition)

**File**: [src/entities/canvas-genome/model/nodes/merge/add_node.ts](src/entities/canvas-genome/model/nodes/merge/add_node.ts)

```typescript
export class AddNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.previous.length > 0) {
            const firstInputShape = this.previous[0].GetOutputShape()
            this.inputShape = [...firstInputShape]
            this.outputShape = [...firstInputShape]  // Output = input shape
        } else {
            this.outputShape = [...this.inputShape]
        }
    }

    // **CRITICAL**: Override standard connection validation
    public CanAcceptConnectionFrom(node: BaseNode, isDisconnectedCheck: boolean = false): boolean {
        if (!super.CanAcceptConnectionFrom(node, isDisconnectedCheck)) return false

        const targetShape = this.GetInputShape()
        const incShape = node.GetOutputShape()

        // All incoming shapes must match EXACTLY (element-wise addition requirement)
        if (targetShape && targetShape.length > 0 && incShape && incShape.length > 0) {
            if (targetShape.length !== incShape.length || 
                !targetShape.every((val, index) => val === incShape[index])) {
                return false
            }
        }
        return true
    }

    protected AddPrev(node: BaseNode): void {
        // Enforce compatibility during connection establishment
        if (this.previous.length > 0) {
            const firstShape = this.previous[0].GetOutputShape()
            const incShape = node.GetOutputShape()
            if (firstShape.length !== incShape.length || 
                !firstShape.every((val, index) => val === incShape[index])) {
                throw new Error("AddNode: Cannot connect. Input shape mismatch!")
            }
        } else {
            this.inputShape = [...node.GetOutputShape()]
        }
        super.AddPrev(node)
    }

    public GetExpectedInputDimensions(): number | "any" {
        return "any"  // Flexible: can add 1D or 3D
    }

    public GetNodeType = (): string => "Add"
    protected _CloneImpl = (): BaseNode => new AddNode()
    public GetIsMerging = (): boolean => true  // **KEY**: Indicates multi-input support
}
```

**Key Difference from BaseNode**:
- `GetIsMerging() = true` → propagation logic skips `SetInputShape` for merge nodes
- Custom `AddPrev()` enforces shape compatibility
- Multiple previous nodes allowed (standard BaseNode limits to 1)

---

### 2.4 FlattenNode (Reshape Layer)

**File**: [src/entities/canvas-genome/model/nodes/layers/flatten_node.ts](src/entities/canvas-genome/model/nodes/layers/flatten_node.ts)

```typescript
export class FlattenNode extends BaseNode {
    constructor() {
        super()
        this.inputShape = new Array<number>(3)  // Expects 3D [H, W, C]
    }

    protected CalculateOutputShape(): void {
        // Flatten [H, W, C] → [H*W*C]
        this.outputShape = [this.inputShape[0] * this.inputShape[1] * this.inputShape[2]]
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 3  // Must have 3D input
    }

    public GetOutputDimensions(): number | "any" {
        return 1  // Always outputs 1D
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {}  // No parameters
        })
    }

    GetResources(dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }  // Pure reshape, no computation
    }

    public GetNodeType = (): string => "Flatten"
    protected _CloneImpl = (): BaseNode => new FlattenNode()
    public GetIsMerging = (): boolean => false
}
```

**Use Case**: Bridge from convolutional feature maps (3D) to dense classification layers (1D).

---

## 3. Backend Burn Translation (src-tauri)

### 3.1 Type Discriminator Enum: NodeDtoJSON

**File**: [src-tauri/src/dtos.rs](src-tauri/src/dtos.rs)

The frontend serializes nodes as JSON using `serde(tag = "node", content = "params")`, creating a discriminated union:

```rust
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    // Layers
    Input {
        output_shape: Vec<u64>,
    },
    Dense {
        units: u64,
        activation: String,
        use_bias: bool,
    },
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
    Pooling {
        pool_type: String,  // "max" | "avg"
        kernel_size: KernelSizeDto,
        stride: u8,
        padding: u8,
    },
    Flatten {},
    Output {
        input_shape: Vec<u64>,
    },
    
    // Merge nodes
    Add {},
    Concat {},
    
    // Regularization
    Dropout { prob: f64 },
    Dropout2D { prob: f64 },
    BatchNorm { epsilon: f64, momentum: f64 },
    LayerNorm { epsilon: f64 },
    GaussianNoise { std_dev: f64 },
}

#[derive(Deserialize, Clone, Debug)]
pub struct KernelSizeDto {
    pub h: u8,
    pub w: u8,
}
```

**Example JSON for Dense node**:
```json
{"node": "Dense", "params": {"units": 256, "activation": "relu", "use_bias": true}}
```

---

### 3.2 Operation Enum (Internal Compilation Target)

**File**: [src-tauri/src/entities.rs](src-tauri/src/entities.rs)

Each `NodeDtoJSON` variant is translated into an `Operation` enum variant, which references **indices into module storage vectors**:

```rust
pub enum Operation {
    Input(usize),                           // Index into input data
    Dense { dense_idx: usize, activation: String },        // Index into denses Vec
    Conv2D { conv2d_idx: usize, activation: String },      // Index into conv2ds Vec
    MaxPool { pool_idx: usize },                            // Index into max_pools Vec
    AvgPool { pool_idx: usize },                            // Index into avg_pools Vec
    Flatten,                                                // No module needed
    Add,                                                    // No module needed
    Concat,                                                 // No module needed
    Dropout { dropout_idx: usize },                         // Index into dropouts Vec
    BatchNorm { batch_norm_idx: usize },                    // Index into batch_norms Vec
    LayerNorm { layer_norm_idx: usize },                    // Index into layer_norms Vec
    Dropout2D { dropout_2d_idx: usize },                    // Index into dropouts Vec
    GaussianNoise { std_dev: f64 },                         // Carries parameter directly
    Output(usize),                                          // Index into output data
}

pub struct GraphModel<B: Backend> {
    pub conv2ds: Vec<Conv2d<B>>,
    pub denses: Vec<Linear<B>>,
    pub max_pools: Vec<MaxPool2d>,
    pub avg_pools: Vec<AvgPool2d>,
    pub dropouts: Vec<Dropout>,
    pub batch_norms_2d: Vec<BatchNorm<B>>,
    pub batch_norms_4d: Vec<BatchNorm<B>>,
    pub layer_norms: Vec<LayerNorm<B>>,
    pub execution_plan: Ignored<Vec<Instruction>>,
    pub use_counts: Vec<usize>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
}
```

---

### 3.3 Translation: NodeDtoJSON → Burn Modules

The compilation process (in `GraphModel::new()`) iterates through nodes in topological order and creates Burn modules:

#### Dense → Linear Example

```rust
NodeDtoJSON::Dense {
    units,
    activation,
    use_bias,
} => {
    let mut actual_units = *units as usize;
    let mut final_activation = activation.clone();

    // Auto-conversion: Last layer softmax → linear for CrossEntropy
    if let Some(&out_idx) = connects_to_output.get(&node_id) {
        if final_activation == "softmax" && actual_units > 1 {
            println!("Auto-converting Output Dense activation from 'softmax' to 'linear'");
            final_activation = "linear".to_string();
        }
    }

    let prev_shape = &shape_cache[inputs_for_node[0]];
    let d_input = prev_shape[0];
    
    // Create Burn Linear layer: (input_dim, output_dim)
    let linear = LinearConfig::new(d_input, actual_units)
        .with_bias(*use_bias)
        .init(device);  // Initialize weights
    
    let dense_idx = denses.len();
    denses.push(linear);
    
    (
        Operation::Dense { dense_idx, activation: final_activation },
        vec![actual_units],  // Output shape
    )
}
```

**Key Translation Insights**:
- **Frontend "units" = Burn "output_dim"**
- **Input shape is inferred** from topological predecessor
- **Burn Linear** expects inputs of shape `[batch_size, input_dim]` → outputs `[batch_size, output_dim]`

#### Conv2D → Conv2d Example

```rust
NodeDtoJSON::Conv2D {
    filters,
    kernel_size,
    stride,
    padding,
    dilation,
    use_bias,
    activation,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    let in_channels = prev_shape[0];      // [C, H, W] Burn format
    let h_in = prev_shape[1];
    let w_in = prev_shape[2];
    let k_h = kernel_size.h as usize;
    let k_w = kernel_size.w as usize;

    let mut actual_filters = *filters as usize;

    // Create Burn Conv2d: [in_channels, out_channels] kernel shape + [k_h, k_w]
    let conv = Conv2dConfig::new([in_channels, actual_filters], [k_h, k_w])
        .with_stride([*stride as usize; 2])
        .with_padding(PaddingConfig2d::Explicit(*padding as usize, *padding as usize))
        .with_dilation([*dilation as usize; 2])
        .with_bias(*use_bias)
        .init(device);

    let conv_idx = conv2ds.len();
    conv2ds.push(conv);

    // Calculate output shape using same formula as frontend
    let h_out = (h_in + 2*(*padding as usize) - (*dilation as usize)*(k_h - 1) - 1) 
                / (*stride as usize) + 1;
    let w_out = (w_in + 2*(*padding as usize) - (*dilation as usize)*(k_w - 1) - 1) 
                / (*stride as usize) + 1;

    (
        Operation::Conv2D { conv2d_idx: conv_idx, activation: activation.clone() },
        vec![actual_filters, h_out, w_out],  // Output shape [C, H, W]
    )
}
```

**Burn Format Note**: Burn uses `[Channels, Height, Width]` (CHW) while frontend uses `[Height, Width, Channels]` (HWC).

#### Flatten → Identity Example

```rust
NodeDtoJSON::Flatten {} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    (Operation::Flatten, vec![prev_shape.iter().product()])
}
```

No Burn module created; flattening is done in `forward()` via `tensor.flatten::<2>(1, 3)`.

---

### 3.4 Forward Pass (Runtime Execution)

The `GraphModel::forward()` method executes the compiled computation graph:

```rust
pub fn forward(&self, inputs: &[DynamicTensor<B>]) -> Vec<DynamicTensor<B>> {
    let mut memory: Vec<Option<DynamicTensor<B>>> = vec![None; self.use_counts.len()];
    let mut remaining_uses = self.use_counts.clone();

    for instr in &self.execution_plan.0 {
        match &instr.op {
            Operation::Dense { dense_idx, activation } => {
                if let DynamicTensor::Dim2(x) = consume!(instr.input_ids[0]) {
                    let linear = &self.denses[*dense_idx];
                    let mut out = linear.forward(x);
                    
                    // Apply activation
                    out = match activation.as_str() {
                        "relu" => burn::tensor::activation::relu(out),
                        "leaky_relu" => burn::tensor::activation::leaky_relu(out, 0.01),
                        "softmax" => burn::tensor::activation::softmax(out, 1),
                        "sigmoid" => burn::tensor::activation::sigmoid(out),
                        "linear" | "none" => out,
                        _ => out,
                    };
                    DynamicTensor::Dim2(out)
                }
            }
            Operation::Conv2D { conv2d_idx, activation } => {
                if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                    let conv = &self.conv2ds[*conv2d_idx];
                    let mut out = conv.forward(x);
                    out = match activation.as_str() {
                        "relu" => burn::tensor::activation::relu(out),
                        "leaky_relu" => burn::tensor::activation::leaky_relu(out, 0.01),
                        _ => out,
                    };
                    DynamicTensor::Dim4(out)
                }
            }
            Operation::Flatten => {
                if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                    DynamicTensor::Dim2(x.flatten::<2>(1, 3))
                }
            }
            // ... more operations
        }
    }
}
```

---

## 4. Polymorphism Pattern

### 4.1 Interface Definition (BaseNode)

All nodes implement the **BaseNode interface**:

```typescript
abstract class BaseNode {
    // Graph structure (polymorphic — implemented by all nodes)
    public previous: BaseNode[]
    public next: BaseNode[]
    public id: string
    
    // Abstract contract — MUST be implemented
    abstract GetNodeType(): string              // Type discriminator
    abstract GetIsMerging(): boolean            // Merge vs. single-input
    abstract GetExpectedInputDimensions(): number | "any"  // Input constraint
    abstract GetOutputDimensions(): number | "any"        // Output constraint
    abstract GetInfo(): string                  // Serialization
    abstract GetResources(dtype: number): ResourceCriteria
    protected abstract CalculateOutputShape(): void
    protected abstract Mutate(mutation_options: Map<string, number>): void
    protected abstract _CloneImpl(): BaseNode
}
```

### 4.2 Method Overriding Patterns

#### Pattern 1: Dimension Validation (BaseNode → Merge Nodes)

**BaseNode** (default): Single input allowed
```typescript
public CanAcceptConnectionFrom(node: BaseNode, isDisconnectedCheck: boolean = false): boolean {
    if (!this.GetIsMerging()) {
        if (this.previous.length >= 1) {  // ← Can only have 1 input
            return false
        }
    }
    // ... other checks
}
```

**AddNode** (override): Multiple inputs with shape matching
```typescript
public CanAcceptConnectionFrom(node: BaseNode, isDisconnectedCheck: boolean = false): boolean {
    if (!super.CanAcceptConnectionFrom(node, isDisconnectedCheck)) return false

    // All inputs must have identical shapes for element-wise add
    const targetShape = this.GetInputShape()
    const incShape = node.GetOutputShape()
    
    if (targetShape.length !== incShape.length || 
        !targetShape.every((val, index) => val === incShape[index])) {
        return false
    }
    return true
}
```

#### Pattern 2: Shape Calculation Customization

**DenseNode**:
```typescript
protected CalculateOutputShape(): void {
    this.outputShape = [this.units]  // Simple: just output dimension
}
```

**Conv2DNode**:
```typescript
protected CalculateOutputShape(): void {
    const hOut = Math.floor(
        (this.inputShape[0] + 2*this.padding - this.dilation*(this.kernelSize.h - 1) - 1) 
        / this.stride + 1
    )
    const wOut = Math.floor(
        (this.inputShape[1] + 2*this.padding - this.dilation*(this.kernelSize.w - 1) - 1) 
        / this.stride + 1
    )
    this.outputShape = [hOut, wOut, this.filters]  // Complex: formula-based
}
```

**AddNode**:
```typescript
protected CalculateOutputShape(): void {
    if (this.previous.length > 0) {
        // Output = first input (element-wise add)
        const firstInputShape = this.previous[0].GetOutputShape()
        this.outputShape = [...firstInputShape]
    }
}
```

#### Pattern 3: Mutation Strategy

Different nodes customize evolutionary mutation:

**DenseNode**:
```typescript
protected Mutate(mutation_options: Map<string, number>): void {
    if (Math.random() <= (mutation_options.get("dense_units") || -1)) {
        this.units = Math.pow(2, RandomizeInteger(4, 12))  // Power-of-2 exploration
    }
    if (Math.random() <= (mutation_options.get("dense_activation") || -1)) {
        this.activation = DenseNode.activationFunctions[RandomizeInteger(0, 2)]
    }
}
```

**Conv2DNode**:
```typescript
protected Mutate(mutation_options: Map<string, number>): void {
    if (Math.random() <= (mutation_options.get("conv2d_filters") || -1)) {
        this.filters = 4 * RandomizeInteger(4, 16)  // Multiples of 4
    }
    if (Math.random() <= (mutation_options.get("conv2d_kernel_size") || -1)) {
        const kernelSize = 1 + 2 * RandomizeInteger(0, 3)  // Odd sizes only
        this.kernelSize = { h: kernelSize, w: kernelSize }
    }
}
```

**FlattenNode**:
```typescript
protected Mutate(_mutation_options: Map<string, number>): void {
    // No mutation (pure reshape)
}
```

---

## 5. Serialization/Deserialization Pipeline

### 5.1 Frontend Serialization

**File**: [src/entities/canvas-genome/lib/serializeGenome.ts](src/entities/canvas-genome/lib/serializeGenome.ts)

```typescript
export const serializeGenome = async (genome: Genome): Promise<string> => {
    const nodes: BaseNode[] = [];
    const nodeIndexes: Map<BaseNode, number> = new Map();
    const nodesToCheck: BaseNode[] = [...genome.inputNodes];

    let output: string = "";
    let nodeCounter = 0;
    
    // BFS to collect all nodes
    while (nodesToCheck.length > 0) {
        const currentNode = nodesToCheck.shift()!;
        if (nodeIndexes.get(currentNode) != undefined) continue;

        nodes.push(currentNode);
        nodeIndexes.set(currentNode, nodeCounter);
        nodeCounter++
        nodesToCheck.push(...currentNode.previous, ...currentNode.next);
        
        // Serialize node to JSON line (polymorphic call to GetInfo)
        output += currentNode.GetInfo() + "\n";
    }

    output += "CONNECTIONS\n";

    // Serialize edges as index pairs
    for (let node of nodes) {
        const currentIndex = nodeIndexes.get(node);
        for (let nextNode of node.next) {
            output += `${currentIndex} ${nodeIndexes.get(nextNode)!}\n`
        }
    }

    return output;
}
```

**Example Output**:
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

---

### 5.2 Frontend Deserialization

**File**: [src/entities/canvas-genome/lib/deserializeGenome.ts](src/entities/canvas-genome/lib/deserializeGenome.ts)

```typescript
export const deserializeGenome = async (genomeStr: string): Promise<{
    nodes: BaseNode[],
    genome: Genome,
}> => {
    const rows = genomeStr.split("\n");
    const nodes: BaseNode[] = [];

    let rowIndex = 0;
    
    // Parse nodes: switch on "node" field (type discriminator)
    while (rows[rowIndex] != "CONNECTIONS" && rowIndex < rows.length) {
        const obj = JSON.parse(rows[rowIndex]);
        rowIndex++;

        switch (obj.node) {
            case "Input":
                const outputShape = obj.params.output_shape;
                nodes.push(new InputNode(outputShape));
                break;
            case "Dense":
                const { units, activation, use_bias } = obj.params;
                nodes.push(new DenseNode(units, activation, use_bias));
                break;
            case "Conv2D":
                const { filters, kernel_size, stride, padding, dilation, use_bias: bias, activation: act } = obj.params;
                nodes.push(new Conv2DNode(filters, kernel_size, stride, padding, dilation, bias, act || 'relu'));
                break;
            // ... more cases
            default:
                throw new Error("Wrong node type");
        }
    }

    // Parse edges
    rowIndex++;
    for (; rowIndex < rows.length - 1; rowIndex++) {
        const indexes = rows[rowIndex].split(" ");
        const fromNodeIndex = Number.parseInt(indexes[0]);
        const toNodeIndex = Number.parseInt(indexes[1]);
        nodes[fromNodeIndex].AddNext(nodes[toNodeIndex]);
    }

    // Extract input/output nodes
    const inputNodes: BaseNode[] = [];
    const outputNodes: BaseNode[] = [];
    for (let node of nodes) {
        if (node.previous.length == 0) inputNodes.push(node);
        if (node.next.length == 0) outputNodes.push(node);
    }

    return { nodes, genome: new Genome(inputNodes, outputNodes) };
}
```

---

### 5.3 Backend Deserialization & Compilation

The Rust backend receives the **same format** and parses it into `NodeDtoJSON`:

```rust
for line in raw_data.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
    if line == "CONNECTIONS" {
        parsing_connections = true;
        continue;
    }
    if parsing_connections {
        // Parse edge indices
        let parts: Vec<usize> = line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        edges.push((parts[0], parts[1]));
    } else {
        // Parse node: serde deserializes JSON → NodeDtoJSON enum
        let config: NodeDtoJSON = serde_json::from_str(line)?;
        configs.push(config);
    }
}
```

Then **GraphModel::new()** translates each `NodeDtoJSON` variant into a `Burn` module and executes in topological order.

---

## 6. Complete Examples

### Example 1: Dense Layer (Start to Finish)

#### Frontend Construction
```typescript
const denseNode = new DenseNode(256, "relu", true);
denseNode.SetInputShape([784]);  // From flattened 28×28 image
denseNode.PropagateShapeUpdate();  // Outputs [256]
```

#### Serialization
```json
{"node":"Dense","params":{"units":256,"activation":"relu","use_bias":true}}
```

#### Backend Compilation (Rust)
```rust
// serde deserializes JSON
let config = NodeDtoJSON::Dense { units: 256, activation: "relu", use_bias: true };

// Translation
let linear = LinearConfig::new(784, 256)  // 784 = flattened input shape
    .with_bias(true)
    .init(device);
denses.push(linear);

// Operation enum
Operation::Dense { dense_idx: 0, activation: "relu" }
```

#### Forward Pass
```rust
Operation::Dense { dense_idx, activation } => {
    if let DynamicTensor::Dim2(x) = consume!(input) {  // [batch, 784]
        let linear = &self.denses[*dense_idx];
        let mut out = linear.forward(x);  // [batch, 256]
        out = relu(out);
        DynamicTensor::Dim2(out)
    }
}
```

---

### Example 2: Conv2D Layer Mutation & Serialization

#### Initial State
```typescript
const convNode = new Conv2DNode(
    32,                    // filters
    { h: 3, w: 3 },       // kernel size
    1,                     // stride
    1,                     // padding
    1,                     // dilation
    true,                  // use_bias
    "relu"
);
convNode.SetInputShape([32, 28, 28]);  // [H, W, C]
convNode.PropagateShapeUpdate();       // Outputs [28, 28, 32] (due to padding=stride)
```

#### Mutation (Evolutionary step)
```typescript
const mutation_options = new Map([["conv2d_filters", 0.5]]);
if (Math.random() < 0.5) {
    convNode.filters = 4 * RandomizeInteger(4, 16);  // e.g., becomes 64
    convNode.CalculateOutputShape();  // Still [28, 28, 64]
}
```

#### Serialization
```json
{"node":"Conv2D","params":{"filters":64,"kernel_size":{"h":3,"w":3},"stride":1,"padding":1,"dilation":1,"use_bias":true,"activation":"relu"}}
```

#### Backend Compilation
```rust
let conv = Conv2dConfig::new([32, 64], [3, 3])  // in_channels=32, out_channels=64
    .with_stride([1; 2])
    .with_padding(PaddingConfig2d::Explicit(1, 1))
    .with_dilation([1; 2])
    .with_bias(true)
    .init(device);
```

---

### Example 3: AddNode (Merge with Shape Validation)

#### Construction
```typescript
const add = new AddNode();
const dense1 = new DenseNode(128, "relu", true);
const dense2 = new DenseNode(128, "relu", true);

dense1.SetInputShape([784]);
dense1.PropagateShapeUpdate();  // [128]

dense2.SetInputShape([784]);
dense2.PropagateShapeUpdate();  // [128]

// Connect both to add node
dense1.AddNext(add);  // ✓ Succeeds: [128] matches
dense2.AddNext(add);  // ✓ Succeeds: [128] matches

// Try invalid connection (would fail)
const dense3 = new DenseNode(256, "relu", true);
dense3.SetInputShape([784]);
dense3.PropagateShapeUpdate();  // [256]
dense3.AddNext(add);  // ✗ Throws: shape [256] ≠ [128]
```

#### Serialization
```json
{"node":"Dense","params":{"units":128,"activation":"relu","use_bias":true}}
{"node":"Dense","params":{"units":128,"activation":"relu","use_bias":true}}
{"node":"Add","params":{}}
CONNECTIONS
0 2
1 2
```

#### Backend Execution
```rust
Operation::Add => {
    match consume!(input_0) {
        DynamicTensor::Dim2(mut sum) => {
            for &in_id in [1..] {  // Second input
                if let DynamicTensor::Dim2(t2) = consume!(in_id) {
                    sum = sum + t2;  // Element-wise addition
                }
            }
            DynamicTensor::Dim2(sum)
        }
    }
}
```

---

## 7. Key Patterns Summary

| Pattern | Purpose | Example |
|---------|---------|---------|
| **Type Discriminator** | Runtime node type identification | `GetNodeType() → "Dense"` |
| **Abstract Methods** | Define polymorphic contracts | `CalculateOutputShape()`, `GetInfo()` |
| **Method Overriding** | Customize behavior per node type | `AddNode.CanAcceptConnectionFrom() override` |
| **Shape Propagation** | Cascading output shape updates | `PropagateShapeUpdate(visited)` recursive call |
| **Merge Nodes** | Multi-input handling | `GetIsMerging() → true`, custom `AddPrev()` |
| **Resource Estimation** | Fitness/parsimony pressure | `GetResources()` returns flash/ram/macs |
| **Cloning** | Deep copy for genetic operators | `Clone()` preserves parameters, new UUID |
| **Serialization** | JSON → Genome persistence | `GetInfo()` → JSON, deserialize via switch(node) |
| **Backend Translation** | Frontend JSON → Burn modules | `NodeDtoJSON` enum → `Operation` enum → Burn modules |

---

## 8. Known Issues & Gotchas

⚠️ **Merge Node Shape Bugs**:
- `AddNode` does **not validate compatibility** during `CheckCompability()` (only at connection time)
- `Concat2DNode` assumes shape length is always 3; crashes on mismatched dimensions

⚠️ **Adapter Generation**:
- Automatic adapters can inflate graphs; tune parsimony coefficient

⚠️ **Burn Format Mismatch**:
- Frontend: `[H, W, C]` (HWC)
- Burn internal: `[C, H, W]` (CHW)
- Conversion happens at Input node initialization

---

This completes the technical deep dive into the neural network node architecture!
