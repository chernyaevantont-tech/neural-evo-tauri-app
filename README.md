# Neural Network Architecture Visual Editor

Modern visual SVG editor for creating and editing neural network architectures within an evolutionary search system. Built with React, TypeScript, and Tauri, with a Rust backend powered by the [burn](https://burn.dev/) ML framework for model compilation and training.

## ✨ Features

### Modern UI Design
- **Dark Theme**: Professional VS Code-inspired dark theme with a curated color palette
- **Custom Title Bar**: Frameless window with custom minimize/maximize/close controls
- **Intuitive Interface**: Collapsible side menu with Layers and Genomes tabs, info panel, and an infinite canvas
- **Responsive Canvas**: Smooth zooming (cursor-centered), panning, and node drag-and-drop
- **Visual Feedback**: Highlighted connections, selection states, and genome validity indicators

### Node Management
- **Full Configuration**: Each node is created with customizable parameters through a modal dialog
- **Live Editing**: Modify node parameters; the system recreates the node instance with updated params
- **Connection Preservation**: Compatible connections are automatically restored after editing
- **Parameter Validation**: Input validation for all node parameters

### Node Types (Layers)

**Layer Nodes:**
- **Input** — Input layer with configurable shape (e.g., `28×28×3` for RGB images)
- **Dense** — Fully connected layer with units, activation (`relu` / `leaky_relu` / `softmax`), and bias options
- **Conv2D** — Convolutional layer with filters, kernel size, stride, padding, dilation
- **Pooling** — Max/Average pooling with kernel size, stride, and padding
- **Flatten** — Flattens multi-dimensional input to a 1D vector
- **Output** — Output layer with configurable shape

**Merge Nodes:**
- **Add** — Element-wise addition (residual / skip connections)
- **Concat2D** — Channel-wise concatenation

### Evolutionary Operations
- **Genome Breeding**: Cross two genomes by extracting a random subgraph from one genome and splicing it into a compatible insertion point in another
- **Random Subgenome Extraction**: Highlight a random connected subgraph within a genome for evolutionary analysis
- **Automatic Adapter Creation**: When shapes are incompatible during breeding, the system creates adapter layers (Dense, Conv2D, Pooling) to bridge the gap

### Rust Backend (burn ML framework)
- **GraphModel** — Universal directed-acyclic-graph neural network model compiled from a genome description
- **Topological Execution** — Forward pass follows a BFS-sorted topological order with reference-counted tensor memory management
- **Dynamic Tensor Support** — Supports both 2D (Dense) and 4D (Conv/Pooling) tensor paths within a single graph
- **Weight Initialization** — Kaiming Normal initialization for `Dense` layers to prevent vanishing gradients
- **Hardware Acceleration** — Configurable `Wgpu` backend for GPU-accelerated tensor operations
- **Training Pipelines**:
  - Full loop via burn's `SupervisedTraining` with `Adam` optimizer, `ConstantLr` scheduler, and `LossMetric` tracking
  - Manual lightweight loop (`train_simple`) supporting Train/Validation/Test data splits, shuffling, and per-epoch metrics (Loss & Accuracy) logging
- **Loss Functions** — Automatic loss selection: `CrossEntropyLoss` for classification (Dim2), `MseLoss` for regression/image tasks (Dim4)

### Interaction Features

**Adding Nodes:**
1. Open the side menu (☰ icon)
2. Switch to the **Layers** tab
3. Click the corresponding button (+ Input, + Dense, + Conv2D, etc.)
4. Configure parameters in the modal dialog
5. Click **Create** to place the node on the canvas

**Editing Nodes:**
1. Right-click on a node → select **Edit Node** from the context menu
2. Modify parameters in the dialog
3. Click **Update** — a new instance is created and compatible connections are restored

**Moving Nodes:**
Click and drag nodes to reposition them on the canvas.

**Connecting Nodes:**
1. Hold **Shift** key
2. Click on the source node, then click on the target node
3. The system validates shape compatibility via `CheckCompability()`

**Deleting:**
- Right-click on a node → **Delete Node**
- Right-click on a connection → **Delete Connection**

**Copying Nodes:**
- Right-click on a node → **Copy Node** to duplicate it

### Canvas Controls

| Action | Control |
|--------|---------|
| Pan | Right-click drag |
| Zoom | Mouse wheel (cursor-centered) |
| Select | Left-click on nodes/connections |
| Connect | Shift + click source → target |
| Context Menu | Right-click on nodes/connections/genomes |

### Genome Operations

**Loading Genomes:**
- Switch to the **Genomes** tab → click **Load Genome**
- Nodes are automatically laid out using a force-directed algorithm
- File format: `.evog`

**Saving Genomes:**
- Click the **Save** button on a genome card in the side panel
- Genome is serialized and saved via Tauri file dialog

**Breeding Genomes:**
- Start breeding from one genome's context menu
- Select a second genome to cross-breed
- The system extracts subgenomes, finds insertion points, and creates adapters as needed

**Genome Validation:**
- **Valid** (green): All input nodes are `InputNode`, all output nodes are `OutputNode`, graph is connected
- **Invalid** (red): Missing proper input/output nodes or disconnected graph

## 🏗️ Architecture

The project follows **Feature-Sliced Design (FSD)** architecture with a **Tauri 2** backend:

### Frontend (React + TypeScript)

```
src/
├── app/                        # Application layer
│   ├── App.tsx                 # Root component with React Router
│   ├── App.css                 # Global styles and resets
│   └── App.module.css          # App-level CSS modules
│
├── pages/                      # Page-level components
│   └── network-editor-page/
│       ├── NetworkEditorPage.tsx   # Main page layout (TitleBar + SideMenu + Canvas + SidePanel)
│       ├── NetworkEditorPage.module.css
│       └── hooks.ts            # Page-level hooks (keyboard events, window resize)
│
├── widgets/                    # Composite UI blocks
│   ├── network-canvas/         # SVG canvas with node/connection rendering
│   │   ├── NetworkCanvas.tsx   # Canvas component (zoom, pan, drag, context menus)
│   │   ├── hooks.ts            # Canvas-specific hooks
│   │   ├── NodeContextMenu/    # Right-click menu on nodes
│   │   ├── ConnectionContextMenu/
│   │   └── GenomContextMenu/
│   ├── side-menu/              # Collapsible left toolbar (Layers/Genomes tabs)
│   │   └── SideMenu.tsx
│   ├── side-panel/             # Right info panel (node info, genome list)
│   │   └── SidePanel.tsx
│   └── title-bar/              # Custom window title bar (minimize, maximize, close)
│       └── TitleBar.tsx
│
├── features/                   # Feature-specific logic (FSD features)
│   ├── add-node/               # Node creation toolbar + config modal
│   ├── edit-node/              # Node editing flow + modal
│   ├── copy-node/              # Node duplication
│   ├── delete-node/            # Node deletion
│   ├── connect-nodes/          # Shift+click connection logic
│   ├── delete-connection/      # Connection removal
│   ├── dragging-move-node/     # Drag-and-drop node positioning
│   ├── select-canvas-entity/   # Selection state management
│   ├── canvas-panning/         # Pan and zoom handlers
│   ├── resize-canvas/          # Canvas dimension tracking
│   ├── genome-save-load/       # Save/Load via Tauri IPC + file dialogs
│   ├── get-subgenome/          # Random subgenome extraction
│   ├── breed-genomes/          # Evolutionary crossover
│   └── delete-genome/          # Genome removal
│
├── entities/                   # Domain entities
│   ├── canvas-genome/          # Core genome entity
│   │   ├── model/
│   │   │   ├── genome.ts       # Genome class (breeding, subgenome extraction, adapters)
│   │   │   ├── store.ts        # Zustand + Immer store (nodes, connections, genomes)
│   │   │   ├── types.ts        # VisualNode, VisualGenome, Connection types
│   │   │   └── nodes/          # Node implementations
│   │   │       ├── base_node.ts          # Abstract BaseNode (shape tracking, graph traversal)
│   │   │       ├── types.ts
│   │   │       ├── layers/
│   │   │       │   ├── input_node.ts
│   │   │       │   ├── dense_node.ts
│   │   │       │   ├── conv_node.ts
│   │   │       │   ├── pooling_node.ts
│   │   │       │   ├── flatten_node.ts
│   │   │       │   └── output_node.ts
│   │   │       └── merge/
│   │   │           ├── add_node.ts
│   │   │           └── concatinate_2d_node.ts
│   │   ├── lib/
│   │   │   ├── calculateLayout.ts    # Force-directed auto-layout for loaded genomes
│   │   │   ├── serializeGenome.ts    # Genome → string serialization
│   │   │   └── deserializeGenome.ts  # String → Genome deserialization
│   │   └── ui/
│   │       ├── Node/                 # SVG node rendering
│   │       ├── ConnectionLine/       # SVG connection rendering
│   │       ├── NodeInfoCard/         # Node info display in side panel
│   │       └── GenomeCard/           # Genome list item card
│   └── canvas-state/            # Canvas UI state (selection, panning, zoom, context menus)
│       └── model/store.ts       # Zustand + Immer store
│
├── shared/                     # Shared utilities
│   ├── ui/                     # Reusable UI components
│   │   ├── Button/
│   │   ├── Modal/
│   │   ├── ContextMenu/
│   │   └── Icons/
│   ├── lib/
│   │   ├── theme.ts            # Design system tokens (colors, typography, spacing, shadows)
│   │   └── nodeColors.ts       # Node type → color/label mapping
│   └── styles/
│       └── variables.css       # CSS custom properties (mirrors theme.ts)
│
├── lib/
│   └── random.ts               # Random utility helpers
│
└── main.tsx                    # Application entry point (React DOM render, Immer setup)
```

### Backend (Rust / Tauri 2)

```
src-tauri/
├── src/
│   ├── main.rs                 # Tauri application entry point
│   ├── lib.rs                  # Tauri commands (save_genome, load_genome) + plugin setup
│   ├── dtos.rs                 # NodeDtoJSON enum — serde-serializable node configs
│   └── entities.rs             # burn ML model (GraphModel, training pipeline)
├── Cargo.toml                  # Rust dependencies (burn 0.20, tauri 2, serde, rfd)
└── tauri.conf.json             # Tauri configuration (window, bundling, CSP)
```

### Key Backend Structures (`entities.rs`)

| Structure | Purpose |
|-----------|---------|
| `DynamicTensor<B>` | Enum for 2D and 4D tensors within a single execution graph |
| `Layer<B>` | burn Module enum wrapping Conv2d, Linear, MaxPool2d, AvgPool2d |
| `Operation` | Describes what each node does (Input, Dense, Conv2D, MaxPool, AvgPool, Flatten, Add, Concat, Output) |
| `Instruction` | Links a node ID to its operation and input node IDs |
| `GraphModel<B>` | The main model: holds `layers` + `execution_plan` + reference counts. Implements `TrainStep` and `InferenceStep` |
| `DynamicBatch<B>` | Training batch: multiple input & target tensors |
| `DynamicBatcher<B>` | burn `Batcher` implementation for `DataLoader` |
| `train()` | Full training via burn `SupervisedTraining` + `Learner` (checkpoints, metrics, LR scheduling) |
| `train_simple()` | Lightweight manual loop with `Adam` optimizer (no Learner overhead) |

## 🎨 Design System

### Theme
- **Modern Dark Theme**: VS Code-inspired color scheme
- **Color Palette**: Curated colors for nodes, UI elements, and states
- **Typography**: Segoe UI font family
- **Spacing System**: Consistent scale (xs: 4px → xxl: 24px)
- **Shadow System**: Three depth levels (sm, md, lg) + focus ring

### Node Colors

| Node Type | Color | Hex |
|-----------|-------|-----|
| Input | Green | `#6bcf7f` |
| Dense | Cyan | `#4fc3f7` |
| Conv2D | Orange | `#ff9f43` |
| Pooling | Purple | `#ab47bc` |
| Flatten | Light Green | `#7cb342` |
| Add | Red | `#ef5350` |
| Concat | Pink | `#ec407a` |
| Output | Bright Red | `#ff5252` |

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+
- **Rust** toolchain (rustup, stable channel)
- **Tauri CLI** (`@tauri-apps/cli`)

### Installation

```bash
# Install frontend dependencies
npm install

# Run in development mode (frontend + Tauri desktop app)
npm run tauri dev

# Build for production
npm run tauri build
```

### Development

```bash
# Run frontend only (web mode, no Tauri)
npm run dev

# Build frontend
npm run build

# Check Rust backend compilation
cd src-tauri && cargo check
```

## 🔧 Technology Stack

### Frontend
- **React 19** — UI framework
- **TypeScript 5.8** — Type safety
- **Vite 7** — Build tool and dev server
- **Zustand 5** — Lightweight state management
- **Immer** — Immutable state updates
- **React Router DOM 7** — Client-side routing
- **React Icons** — Icon library (Bootstrap Icons, Heroicons, etc.)
- **UUID** — Unique identifier generation

### Backend
- **Tauri 2** — Desktop application framework (Rust ↔ JS IPC)
- **burn 0.20** — ML framework (Wgpu, ndarray, autodiff, nn, train)
- **rand 0.10** — Random number generation for dataset shuffling
- **serde / serde_json** — JSON serialization
- **rfd** — Native file dialogs (save/load genomes)

### State Management

The application uses **two Zustand stores** with Immer middleware:

- **`useCanvasGenomeStore`** — Domain data: nodes (`Map<string, VisualNode>`), connections, genomes. Operations: add/edit/delete nodes, connect nodes, add/delete genomes.
- **`useCanvasStateStore`** — Canvas UI state: selection, dragging, panning, zoom, context menus, canvas dimensions.

## 📝 Connection Rules

### Regular Layers (Input, Dense, Conv2D, Pooling, Flatten)
- Output shape is automatically calculated based on input shape
- `CalculateOutputShape()` is called when creating connections

### Add Node
- Requires identical input tensor shapes
- Validation: `inputShape == nodeOutputShape`

### Concat2D Node
- Requires identical height and width (H, W)
- Concatenates along channel dimension (C)
- Validation: `inputShape[0] == nodeOutputShape[0] && inputShape[1] == nodeOutputShape[1]`

## 📚 Usage Tips

- **Shift + Click**: Connect two nodes
- **Right Click**: Open context menu for additional options
- **Mouse Wheel**: Zoom in/out (cursor-centered)
- **Right Drag**: Pan the canvas
- **Left Click**: Select nodes or connections
- Genomes are saved in `.evog` format via native file dialogs
- See [PARAMETER_CONFIG_GUIDE.md](./PARAMETER_CONFIG_GUIDE.md) for detailed parameter configuration documentation