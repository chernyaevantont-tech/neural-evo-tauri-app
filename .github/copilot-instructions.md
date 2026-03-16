# Neural Evolution Tauri App — Development Instructions

## Project Overview

A **desktop neural network visual editor** for designing and evolving neural architectures via evolutionary algorithms. Built with **Tauri 2** (Rust + React) enabling cross-platform desktop apps with native performance.

**Core Purpose**: Interactive graph-based neural network designer with evolutionary algorithms (mutations, crossover) and training backend powered by the Burn ML framework.

---

## Tech Stack

| Layer | Tech | Version |
|-------|------|---------|
| **Frontend** | React 19, TypeScript, Vite | Latest |
| **State** | Zustand 5, Immer 11 | Latest |
| **UI Components** | Chart.js 4.5, Custom Canvas | Latest |
| **Desktop** | Tauri 2 | 2.x |
| **Backend ML** | Rust, Burn 0.20 | 0.20 |
| **Concurrency** | Rayon 1.11 | 1.11+ |

---

## Build & Development Commands

```bash
# Frontend development
npm run dev       # Start Vite dev server (port 1420, HMR 1421)
npm run build     # Production build (TypeScript + Vite)
npm run test      # Run Vitest suite (jsdom environment)

# Desktop app development
npm run tauri     # Tauri CLI (must use with dev/build/sign)

# Rust backend
cargo build          # Compile Rust backend (in src-tauri/)
cargo build --release # Release build with optimizations
```

**Development Workflow**: 
- Run `npm run dev` for frontend iteration (HMR enabled)
- Run `npm run tauri dev` in a separate terminal to launch the Tauri window
- Vite watches `src/` automatically; Tauri restarts on backend changes

---

## Architecture: Feature-Sliced Design (FSD)

```
src/
├── app/              # Root app setup + routing
├── entities/         # Domain models (immutable, no feature imports)
│   ├── canvas-genome/   # BaseNode + Genome class, node implementations
│   └── canvas-state/    # Canvas state (pan, zoom, selection)
├── features/         # ~20+ isolated features with own logic
│   ├── add-node/, breed-genomes/, train-genome/
│   ├── edit-node/, delete-node/, connect-nodes/
│   ├── genome-save-load/, genome-library/
│   ├── evolution-manager/, evolution-studio/
│   └── [other mutations, canvas interactions]
├── widgets/          # Composite UI components
│   ├── network-canvas/   # Main graph renderer
│   ├── side-menu/, side-panel/, title-bar/
├── pages/            # Full-page layouts
│   ├── network-editor-page/
│   ├── evolution-studio-page/
│   ├── dataset-manager-page/
│   └── genome-library-page/
├── shared/           # Reusable hooks, context, styles, lib
│   ├── hooks/, context/, lib/, styles/, ui/
└── lib/              # Utilities (random.ts)
```

**Key Principle**: Features are **fully decoupled**; they import from `entities/` and `shared/` but never from other features directly.

---

## State Management: Zustand + Immer

- **Single source of truth**: Canvas genome store in `src/entities/canvas-genome/model/store.ts`
- **Immutability**: All updates wrapped with Immer for safe nested mutations
- **Data Structures**: Maps for O(1) lookups of nodes/connections/genomes
- **No class mutations**: Use `produce()` for updates; avoid direct `.set()` on entities

Example pattern:
```typescript
const store = useCanvasGenomeStore();
store.updateNode(nodeId, (draft) => {
  draft.shape = newShape;  // Immer makes this safe
});
```

---

## Key Patterns & Conventions

### 1. **Graph Operations**

- **Shape Propagation**: Recursive cascade through `.next` nodes when input shape changes
- **Compatibility Checks**: `CheckCompability()` validates acyclicity, fan-in restrictions, shape constraints before mutations
- **Genome Splitting (BFS)**: When connections deleted, disjoint components get new `genomeId` assignments

### 2. **Node Architecture**

- **Abstract `BaseNode`**: Handles shape tracking, graph traversal, input/output validation
- **Layer Implementations**: 
  - **Spatial**: Dense, Conv2D, Pooling, Flatten, BatchNorm, Dropout
  - **Temporal/Sequential** (for time series): Conv1D, LSTM*, GRU*, RNN* (*planned)
- **Merge Nodes**: Add, Concat2D with special multi-input compatibility rules
- **Type Safety**: Strict TypeScript; all parameters explicitly typed
- **Polymorphism Pattern**: All nodes extend `BaseNode`, implement:
  - `GetNodeType()` - returns discriminator string for serialization
  - `CalculateOutputShape()` - forward shape propagation
  - `Mutate()` - evolutionary parameter changes
  - `_CloneImpl()` - deep copy for crossover
  - `GetExpectedInputDimensions()` / `GetOutputDimensions()` - validation

### 3. **Evolution Mechanics**

**Mutations**:
- `AddNode`: Split random edge, insert layer with auto-adapters for shape matching
- `RemoveNode`: Delete node, recreate connections with adapters if needed
- `RemoveSubgraph`: Macro-pruning via DFS/BFS traversal

**Crossover**:
- Random subgraph extraction from parent + insertion into genome with auto-adapters
- Shape-aware insertion prevents invalid graphs

**Bloat Control**:
- Parsimony pressure: `fitness -= α × node_count`
- Hard `MAX_NODES` limit prevents runaway bloat
- **Warning**: Automatic adapters can inflate graphs; tune α accordingly

### 4. **Rust/Burn Backend (src-tauri/)**

- **Model Compilation**: Src-tauri compiles Burn models from genome structs
- **Training Loop**: Adam optimizer, CrossEntropy/MSE loss, stratified dataset splitting
- **Asynchronousness**: Tauri IPC calls are non-blocking; use `.await` patterns
- **Logit Optimization**: Auto-converts final softmax→linear for CrossEntropyLoss (prevents gradient squashing)

### 5. **Canvas Rendering (NetworkCanvas)**

- **WebGL-backed**: Custom canvas rendering in `src/widgets/network-canvas/`
- **Pan/Zoom**: Implemented via `resize-canvas` and `canvas-panning` features
- **Node Positioning**: Stored in canvas-state; dragging updates via `dragging-move-node`

---

## Time Series & Sequential Data Support

### Current Status
- **Conv1D** (1D convolution): ✅ Recommended approach for temporal patterns
  - Input: `[sequence_length, features]` → Output: `[output_length, filters]`
  - Use case: Detecting local temporal patterns, faster than RNN
  - Burn implementation: Conv2D with height=1 + reshape tricks
  
- **LSTM, GRU, RNN**: 🚧 Planned (Burn 0.20 lacks built-in RNN support)
  - Requires manual implementation or custom Burn module
  - See [TIMESERIES_NODE_IMPLEMENTATION_PLAN.md](TIMESERIES_NODE_IMPLEMENTATION_PLAN.md)

### Typical Time Series Network
```
Input [sequence_length, features]
    ↓
Conv1D(filters=32, kernel=5) → [seq_len, 32]
    ↓
Pooling (optional, for downsampling) → [reduced_seq_len, 32]
    ↓
Conv1D(filters=64, kernel=3) → [seq_len', 64]
    ↓
Flatten → [1D vector]
    ↓
Dense(128) → Dense(output_classes) → Output
```

### Shape Requirements
- **Temporal input must be 2D**: `[sequence_length, features]` (NOT 3D batch dimension—handled by Burn backend)
- **Compatible layers**:
  - Conv1D → Conv1D ✅
  - Conv1D → Flatten ✅ (becomes 1D)
  - Flatten → Dense ✅
  - Dense → Dense ✅
- **Incompatible**:
  - Conv1D → Conv2D ❌ (dimension mismatch)
  - Conv2D → Conv1D ❌ (dimension mismatch)

---

## Documented Issues & Gotchas

⚠️ **Known Issues**:

1. **Merge Node Shape Bugs**:
   - `AddNode` doesn't verify multi-input compatibility on merge nodes
   - `Concat2D` calculates output shape from `.next` instead of `.previous` (incorrect in some cases)
   - **Mitigation**: Test crossover ops with Concat/Add layers thoroughly

2. **Adapter Explosion**:
   - Automatic adapters can inflate graphs significantly
   - **Mitigation**: Tune parsimony coefficient α; set reasonable MAX_NODES

3. **No Speciation**:
   - All genomes compete directly (NEAT-style speciation not implemented)
   - **Impact**: Diversity loss over time; consider hybrid approaches for large populations

4. **Dataset Handling**:
   - CSV uploads are processed in frontend; large datasets may block UI
   - **Mitigation**: Implement streaming or backend-side parsing for files >10MB

---

## Development Workflow Tips

### Adding a New Feature
1. Create `src/features/my-feature/` with `index.ts`, `model/`, `ui/`
2. Keep logic in `model/`; UI in `ui/`
3. Import only from `entities/` and `shared/`, never other features
4. Export public API via `index.ts`

### Modifying Node Types
1. Edit `src/entities/canvas-genome/model/node-implementations.ts`
2. Update shape propagation logic in `BaseNode.propagateShape()`
3. Add compatibility checks in `CheckCompability()`
4. Add Rust equivalent in `src-tauri/src/entities.rs`

### Adding a New Layer Type (Step-by-Step)

**Frontend (TypeScript)**:
1. Create `src/entities/canvas-genome/model/nodes/[category]/new_node.ts`
   - Extend `BaseNode`
   - Implement all abstract methods (see pattern in `DenseNode`, `Conv2DNode`)
   - Define parameters as private properties
   - Implement `CalculateOutputShape()` with correct dimension logic
   - Add `GetNodeType()` returning unique string identifier
   - Implement `Mutate()` with realistic parameter ranges
   - Return `_CloneImpl()` with deep copy of all parameters

2. Add tests in `new_node.test.ts`
   - Test shape calculation with various input dimensions
   - Test parameter mutations
   - Test cloning preserves parameters

3. Register in factory/node creation logic:
   - Add case in `createNodeByType()` function
   - Add UI component for parameter editing if complex

**Backend (Rust)**:
1. Add variant to `NodeDtoJSON` enum in `dtos.rs`
   - Use `#[serde(tag = "node", content = "params")]` convention
   - Add all parameters matching TypeScript definition
   - Include `#[serde(default = "...")]` for optional fields

2. Implement compilation in `entities.rs::build()`
   - Match on `NodeDtoJSON::YourNewNode { ... }`
   - Create corresponding Burn module/layer
   - Calculate output shape using same formula as frontend
   - Return `(Operation::YourNewNode { ... }, output_shape)`

3. Implement forward pass in `entities.rs::GraphModel::forward()`
   - Match on `Operation::YourNewNode { ... }`
   - Apply layer to input tensor(s)
   - Store result in `tensors[instr.node_id]`

**Validation Checklist**:
- ✅ Shape calculation matches between frontend and backend
- ✅ Input dimension expectations documented
- ✅ Parameters serializable to/from JSON
- ✅ Compatible with existing compatibility checks
- ✅ Tested with mock genomes


### Running Evolution Simulations
1. Start dev server and Tauri app
2. Design genome in network editor
3. Upload dataset via dataset-manager-page
4. Configure evolution params in evolution-studio-page
5. Click "Start Evolution" → async loop runs with cancellation support

### Debugging Backend Calls
- Check Tauri console output (Tauri window dev tools)
- Use `console.log()` in Rust via `println!()` or `tracing` crate
- Inspect IPC payloads in Tauri dev tools Network tab

---

## Testing Strategy

- **Unit Tests**: Vitest in jsdom environment (no DOM rendering)
- **Integration Tests**: Test features against Zustand store
- **Manual Testing**: Use dev app for canvas interactions, evolution visuals
- **Backend Tests**: Rust tests in `src-tauri/src/` (run via `cargo test`)

Run: `npm run test` or `npm run test -- --watch`

---

## Common Patterns to Reuse

### Using Canvas State
```typescript
const { selection, panOffset, zoom } = useCanvasState();
store.setSelection(nodeId);
```

### Genome Mutations
```typescript
store.addNode(genomeId, sourceNodeId, targetNodeId, newNodeType);
store.deleteNode(genomeId, nodeId);
```

### Evolution Loop Integration
```typescript
const { isRunning, stats } = useEvolutionLoop({
  populationSize, generations, mutationRate, ...
});
// Check `stats` for real-time fitness, node count, etc.
```

---

## Entry Points for Development

| Component | File |
|-----------|------|
| Main Canvas Renderer | [src/widgets/network-canvas/](src/widgets/network-canvas/) |
| Genome Logic & Mutations | [src/entities/canvas-genome/model/](src/entities/canvas-genome/model/) |
| State Store | [src/entities/canvas-genome/model/store.ts](src/entities/canvas-genome/model/store.ts) |
| Evolution Orchestration | [src/features/evolution-studio/model/](src/features/evolution-studio/model/) |
| Rust Backend | [src-tauri/src/](src-tauri/src/) |

---

## Quick Checklist Before Committing

- [ ] No unused imports (strict TypeScript)
- [ ] All node types have shape propagation logic
- [ ] Compatibility checks pass for new mutations
- [ ] Tests pass (`npm run test`)
- [ ] Tauri app runs without errors (`npm run tauri dev`)
- [ ] No direct feature imports across features
