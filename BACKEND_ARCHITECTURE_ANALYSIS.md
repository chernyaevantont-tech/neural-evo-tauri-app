# Backend Architecture Analysis
## Neural Evolution Tauri App (src-tauri/)

---

## 1. FILE STRUCTURE & MODULES

### Root Directory Structure
```
src-tauri/src/
├── lib.rs                    # Main library entry, Tauri command registration, evolution loop
├── main.rs                   # Minimal - delegates to lib.rs via run()
├── dtos.rs                   # DTO definitions (Node types, Training params, Genealogy)
├── entities.rs               # Neural network model building & training (Burn framework)
├── data_loader.rs            # Dataset loading, batch assembly, stream handling
├── csv_loader.rs             # CSV-specific data loading
├── shape_inference.rs        # Input/output shape propagation logic
├── profiler.rs               # Training profiling (timing, memory, metrics)
├── pareto.rs                 # Multi-objective fitness analysis
├── device_profiles.rs        # Hardware device profiles & constraints
├── genealogy.rs              # Genome lineage tracking (parent->child relationships)
├── weight_io.rs              # Model weight checkpointing
├── zero_cost_proxies.rs      # Fast evaluation scoring without full training
└── orchestrator/             # Training orchestration module
    ├── mod.rs                # Central TrainingOrchestrator
    ├── run_registry.rs       # Tracks active/queued/completed training runs
    ├── scheduler.rs          # Schedules jobs with memory constraints
    └── memory_estimator.rs   # Predicts memory requirements
```

### Key Dependencies (Cargo.toml)
- **Burn 0.20.0**: ML framework with Autodiff, WGPU backend, CUDA support
- **Tauri 2**: Desktop app framework
- **Tokio**: Async runtime (full features)
- **Serde**: JSON serialization
- **Rayon**: Parallel iteration
- **Image 0.25.9**: Image loading for vision datasets
- **CSV 1.4.0**: CSV parsing
- **Chrono**: Timestamps
- **UUID**: Unique ID generation

---

## 2. EVOLUTION LOOP IMPLEMENTATION

### High-Level Flow: `evaluate_population()`

Located in **[lib.rs](lib.rs#L422)** (async Tauri command)

#### **Inputs:**
- `genomes: Vec<String>` - JSON genome descriptions
- `dataset_profile: String` - Profile ID to load
- `batch_size: usize` - Training batch size
- `per_genome_epochs: Vec<usize>` - Epochs per genome
- `dataset_percent: usize` - % of dataset to use (1-100)
- `train_split / val_split / test_split: usize` - Ratios for data partitioning

#### **Evolution Loop Pseudocode:**

```rust
fn evaluate_population(...) -> Result<Vec<EvaluationResult>> {
    // 1. CAPTURE SESSION SNAPSHOT
    let session_snapshot = EVOLUTION_SESSION.load();
    
    // 2. LOAD DATASET PROFILE
    let profile = load_dataset_profiles()
        .find(|p| p.id == dataset_profile);
    let loader = DataLoader::new(profile);
    
    // 3. BUILD TRAIN/VAL/TEST SPLITS
    let mut valid_ids = loader.get_all_valid_sample_ids();
    valid_ids.shuffle();
    
    // Apply dataset_percent filter
    let use_count = (valid_ids.len() * dataset_percent) / 100;
    valid_ids.truncate(use_count);
    
    // STRATIFIED SPLIT: If target is Categorical, split by class
    if let Some(categorical_stream) = find_categorical_target_stream() {
        let groups = group_samples_by_class(categorical_stream);
        for (class_label, samples) in groups {
            // Split class samples: train_ratio / val_ratio / test_ratio
            distribute_samples_to_splits(&mut train_ids, &mut val_ids, &mut test_ids);
        }
        shuffle(train_ids);
        shuffle(val_ids);
        shuffle(test_ids);
    } else {
        // Random split (fallback)
        simple_split(&valid_ids, train_ratio, val_ratio, test_ratio);
    }
    
    // 4. BUILD BATCHES ONCE (reused for all genomes)
    println!("Assembling batches...");
    let train_batches = build_batches!(train_ids, loader);  // Groups by batch_size
    let val_batches = build_batches!(val_ids, loader);
    let test_batches = build_batches!(test_ids, loader);
    
    // 5. GENOME EVALUATION LOOP
    let mut results = Vec::new();
    
    for (i, genome_str) in genomes.iter().enumerate() {
        // CHECK CANCELLATION (session changed = stop_evolution was called)
        if EVOLUTION_SESSION.load() != session_snapshot {
            println!("Evolution cancelled. Aborting remaining genomes.");
            break;
        }
        
        // EMIT PROGRESS EVENT to frontend
        app_handle.emit("evaluating-genome", i);
        
        // CACHE CHECK: Hash(genome + all training params) -> (loss, accuracy)?
        let cache_key = hash(genome_str, dataset_profile, batch_size, epochs, ...);
        if let Some((cached_loss, cached_acc)) = GENOME_EVAL_CACHE.get(&cache_key) {
            eprintln!("CACHE HIT for Genome {}", i);
            results.push(EvaluationResult { loss: cached_loss, accuracy: cached_acc, ... });
            continue;
        }
        
        // BUILD MODEL FROM GENOME
        let model = GraphModel::<Autodiff<Wgpu>>::build(
            genome_str,
            &device,
            Some(&input_overrides),  // e.g., [custom_h, custom_w, custom_c]
            Some(&output_overrides), // e.g., [num_classes for classification]
        );
        
        // RETRY LOOP: Train multiple times with fresh weights
        // (If accuracy < random_chance_threshold, restart with new weights)
        let mut best_loss = 999.0;
        let mut best_acc = 0.0;
        
        for attempt in 0..MAX_RETRIES (3) {
            if EVOLUTION_SESSION.load() != session_snapshot {
                break;  // Cancellation check
            }
            
            // Rebuild model (fresh random weights)
            let mut model = GraphModel::build(...);
            
            // EMIT: "evaluating-genome-start" (frontend clears live charts)
            app_handle.emit("evaluating-genome-start", i);
            
            // TRAINING PHASE (if epochs > 0)
            let epochs = per_genome_epochs[i];
            if epochs > 0 {
                run_eval_pass(
                    &app_handle,
                    &mut model,
                    &train_batches,
                    epochs,
                    learning_rate = 0.001,
                    is_classification,
                    Some(&mut profiler),  // Collects timing & memory metrics
                );
            }
            
            // VALIDATION PHASE
            let (val_loss, val_acc) = run_validation_pass(
                &model,
                &val_batches,
                is_classification,
            );
            
            // TEST/FINAL PHASE
            let (final_loss, final_acc) = if !test_batches.is_empty() {
                run_validation_pass(&model, &test_batches, is_classification)
            } else if !val_batches.is_empty() {
                (val_loss, val_acc)
            } else {
                run_validation_pass(&model, &train_batches, is_classification)  // Fallback
            };
            
            // Keep best attempt
            if attempt == 0 || final_acc > best_acc {
                best_loss = final_loss;
                best_acc = final_acc;
                best_profiler = profiler.finalize();
            }
            
            // EARLY EXIT: If accuracy > RANDOM_CHANCE_THRESHOLD (55%), stop retrying
            if final_acc > 55.0 {
                break;
            }
            
            // AUTOSAVE TO HIDDEN ARCHIVE
            autosave_hidden_genome(genome_id, genome_str, best_loss, best_acc, best_profiler);
        }
        
        // 6. CACHE RESULT (only if above random chance)
        if best_acc > 55.0 {
            GENOME_EVAL_CACHE.insert(cache_key, (best_loss, best_acc));
        }
        
        // 7. EMIT RESULT EVENT to frontend for progressive UI updates
        app_handle.emit("evaluating-genome-result", GenomeResult {
            index: i,
            loss: best_loss,
            accuracy: best_acc,
        });
        
        results.push(EvaluationResult {
            genome_id,
            loss: best_loss,
            accuracy: best_acc,
            profiler: best_profiler,
        });
    }
    
    Ok(results)
}
```

### Key Features:
1. **Session-based Cancellation**: `EVOLUTION_SESSION` counter enables mid-evolution stops
2. **Batch Caching**: Datasets assembled once, reused for all genomes (memory efficient)
3. **Stratified Splits**: If target is categorical, maintains class distribution in train/val/test
4. **Retry Logic**: If model accuracy near random chance, retrain with fresh weights (up to 3x)
5. **Evaluation Cache**: Genome + training params → (loss, accuracy) persists within app lifetime
6. **Hidden Library Autosave**: Automatically archives all evaluated genomes
7. **Progressive Feedback**: `evaluating-genome` and `evaluating-genome-result` events sent to frontend

---

## 3. DTO DEFINITIONS & PATTERNS

### Serialization Format

All DTOs use **Serde** with custom tag/content patterns for polymorphism:

```rust
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    Conv2D { filters, kernel_size, stride, padding, dilation, use_bias, activation },
    Conv1D { filters, kernel_size, stride, padding, dilation, use_bias, activation },
    Dense { units, activation, use_bias },
    Flatten {},
    Input { output_shape: Vec<u64> },
    Output { input_shape: Vec<u64> },
    Pooling { pool_type, kernel_size, stride, padding },
    Add {},
    Concat {},
    Dropout { prob: f64 },
    BatchNorm { epsilon: f64, momentum: f64 },
    LayerNorm { epsilon: f64 },
    Dropout2D { prob: f64 },
    GaussianNoise { std_dev: f64 },
    LSTM { hidden_units, gate_activation, cell_activation, hidden_activation, use_bias },
    GRU { hidden_units, gate_activation, hidden_activation, use_bias, reset_after },
    MultiHeadAttention { n_heads, dropout, quiet_softmax },
    TransformerEncoderBlock { n_heads, d_ff, dropout, activation, norm_first },
}
```

### Genome Format (Wire Protocol)

Genomes are serialized as **multi-line JSON**:
```
{"node": "Input", "params": {"output_shape": [3, 224, 224]}}
{"node": "Conv2D", "params": {"filters": 32, "kernel_size": {"h": 3, "w": 3}, ...}}
{"node": "Dense", "params": {"units": 128, "activation": "relu", ...}}
CONNECTIONS
0 1
1 2
...
```

Each line = one node (sorted by topology). After blank line + "CONNECTIONS", edge list follows.

### Key DTO Structures

#### **EvaluationResult**
```rust
pub struct EvaluationResult {
    pub genome_id: String,
    pub loss: f32,
    pub accuracy: f32,
    pub profiler: Option<TrainingProfiler>,
}
```

#### **TrainingProfiler** (Performance Metrics)
```rust
pub struct TrainingProfiler {
    pub train_start_ms: u64,
    pub first_batch_ms: u64,
    pub train_end_ms: u64,
    pub total_train_duration_ms: u64,
    pub val_start_ms: u64,
    pub val_end_ms: u64,
    pub val_duration_ms: u64,
    pub test_start_ms: u64,
    pub test_end_ms: u64,
    pub test_duration_ms: u64,
    pub peak_active_memory_mb: f32,
    pub peak_model_params_mb: f32,
    pub peak_gradient_mb: f32,
    pub peak_optim_state_mb: f32,
    pub peak_activation_mb: f32,
    pub samples_per_sec: f32,
    pub inference_msec_per_sample: f32,
    pub batch_count: u32,
    pub early_stop_epoch: Option<u32>,
}
```

#### **DatasetProfile** & **DataStream**
```rust
pub struct DatasetProfile {
    pub id: String,
    pub name: String,
    pub source_path: Option<String>,
    pub streams: Vec<DataStream>,
}

pub struct DataStream {
    pub id: String,
    pub alias: String,
    pub role: String,  // "Input" | "Target" | "Ignore"
    pub data_type: DataType,  // Image | Vector | Categorical | Text | TemporalSequence
    pub tensor_shape: Vec<usize>,
    pub num_classes: Option<usize>,
    pub locator: DataLocatorDef,  // How to find files (Glob, Folder, CSV, etc.)
    pub preprocessing: Option<PreprocessingSettings>,
}

pub enum DataType {
    Image,
    Vector,
    Categorical,
    Text,
    TemporalSequence,
}

pub enum DataLocatorDef {
    GlobPattern { pattern: String },           // e.g., "data/**/*.jpg"
    FolderMapping,                              // Use folder as class label
    CompanionFile { path_template: String, parser: String },  // YOLO, COCO, Text
    MasterIndex { index_path, key_field, value_field, has_headers },
    CsvDataset(CsvDatasetDef),
    None,
}
```

#### **GenomeFitnessMetrics**
```rust
pub struct GenomeFitnessMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub adjusted_fitness: Option<f32>,
    pub inference_latency_ms: Option<f32>,
    pub model_size_mb: Option<f32>,
    pub training_time_ms: Option<u64>,
}
```

#### **MutationType** (Genealogy)
```rust
pub enum MutationType {
    Random,
    AddNode { node_type: String, source: String, target: String },
    RemoveNode { node_id: String },
    RemoveSubgraph { node_ids: Vec<String> },
    ParameterMutation { layer_id: String, param_name: String },
    ParameterScale { layer_id: String, scale_factor: f32 },
    Crossover { parent1: String, parent2: String },
}
```

#### **GenomeLibraryEntry** (Stored Genomes)
```rust
pub struct GenomeLibraryEntry {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub input_dims: Vec<usize>,
    pub output_dims: Vec<usize>,
    pub total_nodes: usize,
    pub layer_types: Vec<String>,
    pub best_loss: Option<f32>,
    pub best_accuracy: Option<f32>,
    pub is_hidden: bool,  // Autosaved or visible
    pub source_generation: u32,
    pub parent_genomes: Vec<String>,
    pub fitness_metrics: Option<GenomeFitnessMetrics>,
    pub profiler_data: Option<TrainingProfiler>,
    pub created_at_unix_ms: u64,
}
```

---

## 4. EVOLUTION STATE TRACKING

### Global State Management

**lib.rs** maintains three global lazy-statics:

#### **1. EVOLUTION_SESSION** (u64 counter)
```rust
static EVOLUTION_SESSION: AtomicU64 = AtomicU64::new(0);

// Usage:
let session_snapshot = EVOLUTION_SESSION.load(Ordering::SeqCst);  // Capture at start
if EVOLUTION_SESSION.load() != session_snapshot {
    // stop_evolution() was called -> abort
}
```
- Incremented by `stop_evolution()` command
- Checked at each genome evaluation and every 10 batches during assembly
- Enables **graceful cancellation** without panic

#### **2. GENOME_EVAL_CACHE** (HashMap<u64, (f32, f32)>)
```rust
static GENOME_EVAL_CACHE: LazyLock<Mutex<HashMap<u64, (f32, f32)>>> = 
    LazyLock::new(|| Mutex::new(HashMap::new()));

// Cache key = hash(genome_json + dataset_profile + batch_size + epochs + dataset_percent + train_split + val_split + test_split)
// Value = (loss, accuracy)
```
- **Scope**: Per app session (persists across generations)
- **Granularity**: Exact parameter match required (any diff = cache miss)
- **Gate**: Only cache results > 55% accuracy (filters bad random inits)
- **Use**: Skips re-evaluating identical genome + training conditions

#### **3. GENEALOGY_STORE** (GenealogyGraph)
```rust
static GENEALOGY_STORE: LazyLock<Mutex<GenealogyStore>> = 
    LazyLock::new(|| Mutex::new(GenealogyStore::new()));

// Tracks: genome_id -> GenomeLineageRecord
// Fields: generation, parent_ids, mutation_type, created_at_unix_ms
```
- Built incrementally via Tauri commands: `register_founder`, `register_mutation`, `register_crossover`
- Enables ancestry queries: `get_genealogy()`, `get_ancestors()`, `get_descendants()`

### Per-Genome Runtime State

**During evaluation**, each genome has:
```rust
pub struct GraphModel<B: Backend> {
    // Layer collections
    pub conv1ds: Vec<Conv1d<B>>,
    pub conv2ds: Vec<Conv2d<B>>,
    pub denses: Vec<Linear<B>>,
    pub lstms: Vec<Lstm<B>>,
    pub grus: Vec<Gru<B>>,
    // ... other layer types ...
    
    // Execution metadata
    pub execution_plan: Vec<Instruction>,  // DAG of operations
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}
```

**During training**, state is captured:
```rust
// In run_eval_pass:
let mut optim = AdamConfig::new().init::<B, GraphModel<B>>();
// Adam accumulates: momentum (m), velocity (v), t (timestep)

let mut profiler = ProfilerCollector::new();
profiler.mark_train_start();
// ... training loop ...
profiler.finalize()  // Returns TrainingProfiler with all metrics
```

---

## 5. TAURI COMMANDS REGISTRATION

### **tauri::generate_handler!** Macro

**[lib.rs line 2695](lib.rs#L2695)**

```rust
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // File I/O
            save_genome,
            load_genome,
            pick_folder,
            
            // Testing
            test_neural_net_training,
            test_train_on_image_folder,
            
            // Evolution Core
            evaluate_population,        // Main evaluation loop (async)
            stop_evolution,             // Increment EVOLUTION_SESSION
            
            // Dataset Management
            scan_dataset,
            cache_dataset,
            validate_dataset_profile,
            load_dataset_profiles,
            save_dataset_profiles,
            preview_csv,
            
            // Genome Library
            list_library_genomes,
            list_hidden_library,
            save_to_library,
            delete_from_library,
            load_library_genome,
            unhide_genome,
            delete_hidden_genome,
            export_genome_with_weights,
            has_cached_weights,
            
            // Profiling & Analysis
            compute_zero_cost_score,
            compute_pareto_front,
            
            // Device Profiles
            get_device_profiles,
            validate_genome_for_device,
            apply_device_penalty,
            
            // Genealogy Tracking
            register_founder,
            register_mutation,
            register_crossover,
            get_genealogy,
            get_ancestors,
            get_descendants,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Command Type Signature Pattern

All commands are **async** Tauri functions:

```rust
#[tauri::command]
async fn evaluate_population(
    app_handle: tauri::AppHandle,  // Access to emit events
    genomes: Vec<String>,
    dataset_profile: String,
    // ... more params ...
) -> Result<Vec<EvaluationResult>, String> {
    // Token-based cancellation
    let session_snapshot = EVOLUTION_SESSION.load(Ordering::SeqCst);
    
    // Emit progress events
    app_handle.emit("evaluating-genome", i)?;
    app_handle.emit("evaluating-genome-result", result)?;
    
    Ok(results)
}

#[tauri::command]
async fn stop_evolution() -> Result<(), String> {
    let prev = EVOLUTION_SESSION.fetch_add(1, Ordering::SeqCst);
    println!("Evolution cancellation requested (session {} -> {})", prev, prev + 1);
    Ok(())
}
```

### Event Emissions (Frontend Communication)

During `evaluate_population`, the backend emits Tauri events:

| Event | Payload | Purpose |
|-------|---------|---------|
| `evaluating-genome` | `i: usize` | Genome index started |
| `evaluating-genome-start` | `i: usize` | Clears UI charts before training |
| `evaluating-genome-result` | `{index, loss, accuracy}` | Progressive fitness update |

Frontend listens via:
```typescript
appHandle.listen('evaluating-genome-result', (event) => {
    const { index, loss, accuracy } = event.payload;
    // Update UI charts in real-time
});
```

---

## 6. NEURAL NETWORK BUILDING & TRAINING

### Model Construction: `GraphModel::build()`

**[entities.rs line 247](../src-tauri/src/entities.rs#L247)**

1. **Parse** genome JSON (multi-line format) into node configs
2. **Topological Sort**: Ensures feedforward DAG
3. **Layer Instantiation**: Create Burn layer objects
   - Conv1D/Conv2D → `burn::nn::conv::{Conv1d, Conv2d}`
   - Dense → `burn::nn::Linear`
   - LSTM/GRU → `burn::nn::lstm::{Lstm, Gru}`
   - Pooling → `burn::nn::pool::{MaxPool2d, AvgPool2d}`
4. **Shape Propagation**: Forward-pass shape computation
5. **Execution Plan**: DAG of `Instruction`s (topologically sorted)

### Training Loop: `run_eval_pass()`

**[entities.rs line 1551](../src-tauri/src/entities.rs#L1551)**

```rust
fn run_eval_pass<B: AutodiffBackend>(
    app_handle: &tauri::AppHandle,
    model: &mut GraphModel<B>,
    batches: &[DynamicBatch<B>],
    num_epochs: usize,
    learning_rate: f64,
    is_classification: bool,
    session_counter: &AtomicU64,
    session_snapshot: u64,
    profiler: Option<&mut ProfilerCollector>,
) -> (f32, f32) {
    // Optimizer: Adam (learning_rate = 0.001)
    let mut optim = AdamConfig::new().init::<B, GraphModel<B>>();
    
    // Main loop
    for epoch in 0..num_epochs {
        // Check cancellation
        if session_counter.load() != session_snapshot {
            println!("Training cancelled at epoch {}", epoch);
            break;
        }
        
        let mut train_loss_sum = 0.0;
        let mut train_correct = 0;
        let mut train_total = 0;
        
        for (batch_idx, batch) in batches.iter().enumerate() {
            // Cancellation check every 50 batches
            if batch_idx % 50 == 0 && session_counter.load() != session_snapshot {
                return (final_loss, final_acc);
            }
            
            // Clone inputs/targets (fresh autodiff graph each pass)
            let inputs = batch.inputs.clone();
            let targets = batch.targets.clone();
            
            // Forward pass
            let logits = model.forward(&inputs);  // Produces Dim2 or Dim4 tensors
            
            // Loss computation
            let loss = if is_classification {
                // CrossEntropyLoss: targets must be [batch, 1] (class indices)
                compute_cross_entropy_loss(&logits, &targets)
            } else {
                // MSELoss
                compute_mse_loss(&logits, &targets)
            };
            
            // Backward pass
            let grads = loss.backward();
            optim.step(grads);  // Update weights via Adam optimizer
            
            // Accumulate metrics
            train_loss_sum += loss.clone().into_data().to_vec()[0];
            if is_classification {
                let pred = argmax(logits);
                train_correct += (pred == targets).sum();
            }
            train_total += batch.size();
        }
        
        // Epoch summary
        let avg_loss = train_loss_sum / train_total;
        let accuracy = (train_correct as f32 / train_total as f32) * 100.0;
        println!("Epoch {}: loss={:.4}, acc={:.2}%", epoch, avg_loss, accuracy);
    }
    
    return (final_loss, final_acc);
}
```

### Validation Loop: `run_validation_pass()`

**[entities.rs line 1707](../src-tauri/src/entities.rs#L1707)**

Similar to training but:
- **No backward pass** (inference only)
- **No optimizer updates**
- Returns final (loss, accuracy) for evaluation

---

## 7. ORCHESTRATOR & RUN MANAGEMENT

### TrainingOrchestrator Structure

**[orchestrator/mod.rs](../src-tauri/src/orchestrator/mod.rs)**

```rust
pub struct TrainingOrchestrator {
    pub run_registry: Arc<RwLock<RunRegistry>>,
    pub scheduler: Arc<Scheduler>,
    pub memory_estimator: Arc<MemoryEstimator>,
    pub stopping_policy: Arc<RwLock<StoppingPolicy>>,
}

impl TrainingOrchestrator {
    pub async fn start_training_run(&self, max_parallel_jobs: u32) -> Result<String> {
        let run_id = Uuid::new_v4().to_string();
        let mut registry = self.run_registry.write()?;
        registry.create_run(&run_id, max_parallel_jobs)?;
        Ok(run_id)
    }
    
    pub async fn enqueue_job(&self, run_id: &str, job: TrainingJob) -> Result<()> {
        let mut registry = self.run_registry.write()?;
        registry.enqueue_job(run_id, job)?;
        self.scheduler.attempt_schedule(run_id).await?;
        Ok(())
    }
    
    pub async fn stop_run(&self, run_id: &str) -> Result<()> {
        let mut registry = self.run_registry.write()?;
        registry.mark_run_stopping(run_id)?;
        self.scheduler.cancel_active_jobs(run_id).await?;
        Ok(())
    }
    
    pub fn get_run_status(&self, run_id: &str) -> Result<RunState> {
        let registry = self.run_registry.read()?;
        registry.get_run_state(run_id)
    }
}
```

### RunState & JobTracking

```rust
pub struct RunState {
    pub run_id: String,
    pub status: String,  // "pending" | "running" | "stopping" | "stopped"
    pub created_at_ms: u64,
    pub started_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
    pub queued_jobs: Vec<String>,
    pub active_jobs: Vec<String>,
    pub completed_jobs: Vec<String>,
    pub failed_jobs: Vec<String>,
    pub max_parallel_jobs: u32,
}
```

---

## 8. DATA LOADING PIPELINE

### DataLoader Architecture

**[data_loader.rs](../src-tauri/src/data_loader.rs)**

```rust
pub struct DataLoader {
    pub profile: DatasetProfile,
    pub stream_files: HashMap<String, HashMap<String, PathBuf>>,
    pub stream_classes: HashMap<usize, usize>,  // stream_idx -> num_classes
    pub valid_sample_ids: Vec<String>,
}

impl DataLoader {
    pub fn new(profile: DatasetProfile, app_data_dir: Option<PathBuf>) -> Result<Self> {
        // 1. Validate streams (Input/Target/Ignore roles)
        // 2. For each stream, collect files matching DataLocatorDef
        //    - GlobPattern: glob search
        //    - FolderMapping: use folder name as class
        //    - CsvDataset: parse CSV
        //    - CompanionFile: load via path template
        // 3. Align sample IDs across all streams
        // 4. Calculate num_classes for categorical streams
        Ok(Self { ... })
    }
    
    pub fn load_sample(&self, id: &str, device: &B::Device) 
        -> Result<SampleData<B>> {
        // For each stream in profile:
        // - Read file (based on stream.locator type)
        // - Preprocess (resize, normalize, etc.)
        // - Convert to Tensor
        // - Return sample with stream_tensors: HashMap<stream_idx, Tensor>
        Ok(SampleData { stream_tensors })
    }
}
```

### Batch Assembly Macro

During `evaluate_population`, batches are built via macro:

```rust
macro_rules! build_batches {
    ($ids:expr, $loader:expr, $dev:expr) => {{
        let mut batches = Vec::new();
        for (idx, chunk) in $ids.chunks(batch_size).enumerate() {
            // Check cancellation every 10 chunks
            if idx % 10 == 0 && EVOLUTION_SESSION.load() != session_snapshot {
                return Err("Cancelled during batch assembly".to_string());
            }
            
            // Load samples for this chunk
            for id in chunk {
                match $loader.load_sample(id, $dev) {
                    Ok(sample) => {
                        // Collect tensors by stream index
                        for (stream_idx, tensor) in sample.stream_tensors {
                            batch_inputs[stream_idx].push(tensor);
                        }
                    }
                    Err(e) => eprintln!("[ERROR] Dropped sample '{}': {}", id, e),
                }
            }
            
            // Concatenate tensors into batch
            batches.push(DynamicBatch {
                inputs: concat_dynamic_tensors(collected_inputs),
                targets: concat_dynamic_tensors(collected_targets),
            });
        }
        batches
    }};
}
```

---

## 9. KEY DESIGN PATTERNS

### 1. **Polymorphic Nodes (Serde Tag + Content)**
```rust
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON { ... }
// JSON: {"node": "Conv2D", "params": {...}}
```

### 2. **DynamicTensor Enum**
Handles variable tensor dimensions (2D, 3D, 4D):
```rust
pub enum DynamicTensor<B: Backend> {
    Dim2(Tensor<B, 2>),  // [Batch, Features]
    Dim3(Tensor<B, 3>),  // [Batch, Channels, Length]
    Dim4(Tensor<B, 4>),  // [Batch, Channels, Height, Width]
}
```

### 3. **Session-Based Cancellation**
Atomic counter snapshot enables mid-run abort without panics:
```rust
let session_snapshot = EVOLUTION_SESSION.load();
// ... do work ...
if EVOLUTION_SESSION.load() != session_snapshot {
    // Gracefully exit
}
```

### 4. **Evaluation Cache (Memoization)**
Hash of genome + all training parameters avoids redundant evals:
```rust
let cache_key = hash(genome_str, dataset, batch_size, epochs, ...);
if let Some((loss, acc)) = GENOME_EVAL_CACHE.get(&cache_key) {
    // Return cached result
}
```

### 5. **Profiler Injection (Optional)**
Training functions accept optional profiler to collect metrics without always computing:
```rust
pub fn run_eval_pass(..., profiler: Option<&mut ProfilerCollector>, ...) {
    if let Some(p) = profiler.as_mut() {
        p.mark_train_start();
    }
    // ... training ...
    p.finalize()  // Returns TrainingProfiler
}
```

### 6. **Stratified Dataset Splitting**
If target stream is categorical, maintains class distribution:
```rust
let groups = group_samples_by_class(target_stream);
for (class, samples) in groups {
    // Split each class separately, then merge back
    train_ids.extend(samples[..train_count]);
    val_ids.extend(samples[train_count..]);
}
```

---

## 10. SUMMARY TABLE

| Aspect | Implementation |
|--------|-----------------|
| **Async Runtime** | Tokio (all commands async) |
| **ML Framework** | Burn 0.20 (Autodiff<Wgpu>) |
| **Concurrency Model** | Tauri IPC → tokio::task::spawn_blocking for training |
| **State Management** | Global LazyLock statics (Session counter, Eval cache, Genealogy) |
| **Cancellation** | AtomicU64 session counter (checked every epoch/batch) |
| **Caching** | HashMap<hash, (loss, acc)> persists per session |
| **Dataset Splits** | Stratified (by categorical target) or random fallback |
| **Profiling** | Optional ProfilerCollector tracks timing & memory |
| **Genealogy** | In-memory GenealogyStore with parent→child relations |
| **Model Building** | JSON → AST → Topological sort → Layer instantiation |
| **Training** | Adam optimizer, CrossEntropy/MSE loss, batch-wise updates |
| **Validation** | Inference-only pass (no gradients/optimizer updates) |

---

## CRITICAL FILES FOR MODIFICATION

When implementing new features:

1. **Add Node Type**: Update `dtos.rs` (NodeDtoJSON enum) + `entities.rs` (build + forward methods)
2. **Add Tauri Command**: Define in `lib.rs`, add to `invoke_handler!` macro
3. **Modify Evolution Loop**: Edit `evaluate_population()` in `lib.rs`
4. **Add Dataset Type**: Update `dtos.rs` (DataLocatorDef, DataType) + `data_loader.rs`
5. **Add Genealogy Tracking**: Register via `register_mutation()` / `register_crossover()` in `lib.rs`
6. **Add Profiling**: Extend `TrainingProfiler` in `dtos.rs` + `profiler.rs`

