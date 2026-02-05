# Neural Network Architecture Visual Editor

Modern visual SVG editor for creating and editing neural network architectures within an evolutionary search system. Built with React, TypeScript, and Tauri.

## ✨ Features

### Modern UI Design
- **Dark Theme**: Professional VS Code-inspired dark theme with modern color palette
- **Intuitive Interface**: Clean, organized toolbar and side panels
- **Responsive Canvas**: Smooth zooming, panning, and node manipulation
- **Visual Feedback**: Highlighted connections, selection states, and validity indicators

### Node Management
- **Full Configuration**: Each node is created with customizable parameters through a modal dialog
- **Live Editing**: Modify node parameters with automatic instance recreation
- **Connection Preservation**: Compatible connections are automatically restored after editing
- **Parameter Validation**: Input validation for all node parameters

### Node Types (Layers)

**Layer Nodes:**
- **Input** - Input layer with configurable shape (e.g., 28x28x3 for RGB images)
- **Dense** - Fully connected layer with units, activation (relu/leaky_relu/softmax), and bias options
- **Conv2D** - Convolutional layer with filters, kernel size, stride, padding, dilation
- **Pooling** - Max/Average pooling with kernel size, stride, and padding
- **Flatten** - Flattens multi-dimensional input
- **Output** - Output layer with configurable shape

**Merge Nodes:**
- **Add** - Element-wise addition for residual connections
- **Concat2D** - Channel-wise concatenation

### Interaction Features

**Adding Nodes:**
1. Click the corresponding button in the toolbar (+ Input, + Dense, etc.)
2. Configure parameters in the modal dialog
3. Click **Create** to add the node to the canvas

**Editing Nodes:**
1. Right-click on a node
2. Select **Edit Node** from the context menu
3. Modify parameters in the dialog
4. Click **Update** to apply changes
5. Compatible connections are automatically restored

**Moving Nodes:**
Click and drag nodes to reposition them on the canvas.

**Connecting Nodes:**
1. Hold **Shift** key
2. Click on the source node
3. Click on the target node
4. System validates compatibility using `CheckCompability()`

**Deleting Nodes:**
1. Right-click on a node
2. Select **Delete Node** from the context menu

**Copying Nodes:**
1. Right-click on a node
2. Select **Copy Node** to create a duplicate

### Canvas Controls

- **Pan**: Right-click and drag, or use middle mouse button
- **Zoom**: Mouse wheel to zoom in/out (centers on cursor position)
- **Select**: Left-click on nodes or connections
- **Context Menu**: Right-click on nodes or connections for additional options

### Genome Operations

**Loading Genomes:**
- Click **Load Genome** button to import a genome from file
- Nodes are automatically laid out on the canvas
- Connections are preserved from the saved genome

**Saving Genomes:**
- Click the **Save** button next to a genome in the side panel
- Only valid genomes (with proper Input/Output nodes) can be saved

**Genome Validation:**
- **Valid**: All input nodes are InputNode instances, all output nodes are OutputNode instances
- **Invalid**: Missing proper input/output nodes or disconnected graph
- Validation status is shown with color indicators (green/red)

**Subgenome Extraction:**
- Click **Get Subgenome** to highlight a random subgraph
- Useful for evolutionary operations and graph analysis

## 🏗️ Architecture

The project follows **Feature-Sliced Design (FSD)** architecture for better scalability and maintainability:

```
src/components/
├── types.ts               # Типы для визуальных нод и соединений
├── NodeRenderer.tsx       # Компонент отрисовки отдельной ноды
├── ConnectionRenderer.tsx # Компонент отрисовки соединений

```
src/
├── app/                    # Application layer
│   ├── App.tsx            # Main application component
│   └── App.css            # Global styles
│
├── widgets/               # Composite UI components
│   ├── network-canvas/    # Main canvas widget with node/connection management
│   │   ├── NetworkCanvas.tsx
│   │   └── hooks.ts       # Canvas state management hooks
│   └── side-panel/        # Information panel widget
│       └── SidePanel.tsx
│
├── features/              # Feature-specific logic
│   ├── node-toolbar/      # Node creation and configuration
│   │   ├── NodeToolbar.tsx
│   │   └── NodeConfigForm.tsx
│   └── genome-operations/ # Genome manipulation features
│       └── ContextMenu.tsx
│
├── entities/              # Business entities
│   ├── node/             # Node entity with UI and logic
│   │   └── ui/
│   │       ├── NodeCard.tsx      # Visual node representation
│   │       └── NodeInfoCard.tsx  # Node information display
│   ├── connection/       # Connection entity
│   │   └── ui/
│   │       └── ConnectionLine.tsx
│   └── genome/           # Genome entity
│       └── ui/
│           └── GenomeList.tsx
│
├── shared/               # Shared utilities and components
│   ├── ui/              # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Icons.tsx
│   │   └── Modal.tsx
│   ├── lib/             # Utilities and helpers
│   │   ├── theme.ts     # Theme configuration
│   │   └── nodeColors.ts
│   ├── api/             # API layer
│   │   └── genome.ts
│   └── types/           # Shared TypeScript types
│       └── index.ts
│
├── evo/                 # Evolution logic (not modified)
│   ├── genome.ts
│   ├── types.ts
│   └── nodes/           # Node implementations
│       ├── base_node.ts
│       ├── layers/
│       └── merge/
│
└── saver/              # Serialization logic
    ├── loadGenome.ts
    └── saveGenome.ts
```

### FSD Benefits

- **Separation of Concerns**: Clear boundaries between layers
- **Reusability**: Shared components can be easily reused across features
- **Scalability**: Easy to add new features without affecting existing code
- **Maintainability**: Logical structure makes codebase easier to navigate
- **Team Collaboration**: Different teams can work on different layers independently

## 🎨 Design System

### Theme
- **Modern Dark Theme**: Professional color scheme inspired by VS Code
- **Color Palette**: Carefully selected colors for nodes, UI elements, and states
- **Typography**: Segoe UI font family for consistency
- **Spacing System**: Consistent spacing scale (xs, sm, md, lg, xl, xxl)
- **Shadow System**: Depth through subtle shadows

### Node Colors
- Input: #6bcf7f (Green)
- Dense: #4fc3f7 (Cyan)
- Conv2D: #ff9f43 (Orange)
- Pooling: #ab47bc (Purple)
- Flatten: #7cb342 (Light Green)
- Add: #ef5350 (Red)
- Concat: #ec407a (Pink)
- Output: #ff5252 (Bright Red)

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Rust (for Tauri)

### Installation

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev

# Build for production
npm run tauri build
```

### Development

```bash
# Run frontend only (web mode)
npm run dev

# Build frontend
npm run build
```

## 🔧 Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Tauri 2** - Desktop application framework
- **Vite** - Build tool and dev server
- **UUID** - Unique identifier generation

## 📝 Connection Rules

### Regular Layers (Input, Dense, Conv2D, Pooling, Flatten)
- Output shape is automatically calculated based on input shape
- `CalculateOutputShape()` method is called when creating connections

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

## 🤝 Contributing

The architecture is designed to be extensible. To add new features:

1. **New Node Type**: Add implementation in `src/evo/nodes/`
2. **New Feature**: Create in `src/features/` following FSD principles
3. **New UI Component**: Add to `src/shared/ui/` if reusable, or in relevant feature/widget

## 📄 License

This project is part of a neural architecture search research system.

---

Built with ❤️ using modern web technologies and evolutionary algorithms.
- [v] Зум и панорамирование холста
- [ ] Автоматическая компоновка графа
