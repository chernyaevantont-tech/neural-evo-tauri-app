# Dataset Loading & Configuration System Refactoring Plan

## Executive Summary

The current dataset loading system has accumulated technical debt and architectural issues:
- **Temporal window handling is fragile** (off-by-one errors, misaligned samples)
- **Input/Output shape inference is incomplete** (Output shape from tensorShape instead of numClasses)
- **CSV stream role handling is inconsistent** (features/targets mixed incorrectly)
- **Shape propagation is scattered** (DataLoader, scan_dataset, frontend) with no single source of truth
- **No pre-validation of dataset compatibility** with genome architecture
- **DRY violation**: Shape calculations repeated in backend, frontend, and randomArchitectureGenerator

This plan consolidates dataset handling into coherent, testable components.

---

## I. PROBLEM ANALYSIS

### Current Issues

#### 1. **Temporal Window Sample Count Mismatch**
- **Problem**: Input stream (temporal_window) has N samples, Target stream (row) has M rows
- **Result**: Zero sample intersection when M ≠ N - window_size + 1
- **Root Cause**: No unified alignment strategy; pre-loads try to force alignment ad-hoc
- **Evidence**: 
  ```rust
  // Current hack in data_loader.rs:
  let sample_count = if stream.role == "Target" && temporal_window_count.is_some() {
      temporal_window_count.unwrap()  // Force target count to match input
  } else {
      csv_loader.num_samples
  };
  ```

#### 2. **Output Shape Determined from Wrong Source**
- **Problem**: `extractShapesFromDatasetProfile` uses `targetStreams[0].tensorShape` instead of numClasses
- **Result**: Output nodes created with shape [19639] (total rows) instead of [5] (num_classes)
- **Root Cause**: tensorShape tracks input dimensions, not classification outputs
- **Fix Started**: Added numClasses field, but not consistently used everywhere

#### 3. **CSV Stream Role Mishandling**
- **Problem**: Target streams receive feature_columns instead of empty array
- **Result**: CSV loader tries to parse features as floats for Target → crashes or reads wrong data
- **Flow**: DatasetManagerPage sends wrong config → scan_dataset ignores role → stream initialized incorrectly

#### 4. **Shape Propagation Scattered**
- **Places where shapes are calculated**:
  - `csv_loader.rs`: Determines temporal window count
  - `data_loader.rs`: Aligns stream sample counts
  - `lib.rs` scan_dataset: Returns input_shape in StreamScanReport
  - `DatasetManagerPage.tsx`: Updates profile.streams after scan
  - `randomArchitectureGenerator.ts`: Extracts shapes from streams
  - `extractShapesFromDatasetProfile`: Converts tensorShape → inputShape, numClasses → outputShape
- **Problem**: No single definition of "what is the input shape for this dataset?"
- **Impact**: Changes to shape logic require updates in 5+ places

#### 5. **No Input/Output Validation Before Evolution**
- **Problem**: User can start evolution with:
  - Input stream tensorShape = [50, 12]
  - Output shape = [] (empty)
  - Result: Random architecture generation fails silently
- **Missing**: Pre-flight validation before `startEvolution()`

#### 6. **Node Type Inference Too Simplistic**
- **Current Logic** in `generateRandomArchitecture`:
  ```typescript
  if (inputDimensions === 3) {
      // Image [H, W, C]
      buildConvolutionalPath();
  } else if (inputDimensions === 2) {
      // Sequence [seq_len, features]
      buildSequentialPath();  // Always uses Conv1D/LSTM/GRU
  } else if (inputDimensions === 1) {
      // Dense [features]
      buildDensePath();
  }
  ```
- **Problem**: 
  - No distinction between temporal vs tabular 2D data
  - Conv1D assumes temporal, but could be multi-feature tabular
  - No way to hint "use Conv1D" vs "use Dense" based on DataType
- **Example**: 2D CSV tabular data [num_samples, features] wrongly routed to Conv1D path

#### 7. **Inconsistent CSV Handling Between Scan & Load**
- **scan_dataset**: 
  - Pre-loads CSV once per unique file
  - Detects temporal window count
  - Returns input_shape for Input streams
- **evaluate_population**:
  - DataLoader re-initializes CSV completely
  - May recalculate sample counts differently
  - No caching of pre-computed shapes
- **Result**: Misalignment if temporal logic differs between scan and eval

---

## II. PROPOSED ARCHITECTURE

### Core Principle
**Single Source of Truth**: Each dataset aspect has ONE place where it's computed and cached.

### New Component: `DatasetMetadata`

```typescript
// Frontend: src/features/dataset-manager/model/types/metadata.ts
export interface DataStreamMetadata {
    // Identity
    streamId: string;
    alias: string;
    role: 'Input' | 'Target' | 'Ignore';
    dataType: DataType;
    
    // Shape information (computed during scan)
    tensorShape: number[];      // E.g., [50, 12] for temporal, [features] for vector
    numClasses?: number;        // For Target/classification streams
    
    // Data characteristics
    sampleCount: number;        // Actual sample count after temporal windowing
    discoveredClasses?: string[]; // Unique class labels for categorical
    
    // Temporal-specific
    isTemporalWindow: boolean;
    windowSize?: number;        // Only if isTemporalWindow=true
    
    // CSV-specific
    isCsvDataset?: boolean;
    csvPath?: string;
    featureColumnIndices?: number[]; // Resolved indices, not names
    targetColumnIndex?: number;
}

export interface DatasetMetadata {
    profileId: string;
    profileName: string;
    sourcePath: string;
    
    // All streams with computed metadata
    streams: DataStreamMetadata[];
    
    // Global intersection
    validSampleIds: string[];  // After filtering across all streams
    totalValidSamples: number;
    droppedSampleIds: string[];
    
    // Validation status
    isValid: boolean;           // All streams have valid shapes and samples
    validationErrors: string[]; // Human-readable errors
    
    // Scan timestamp
    scannedAt: string | null;
}
```

### New Component: `ShapeInference` Service

```rust
// Backend: src-tauri/src/shape_inference.rs

pub struct ShapeInference {
    profile: DatasetProfile,
    root_path: PathBuf,
}

impl ShapeInference {
    /// Compute input shape for a single stream
    pub fn infer_input_shape(
        stream: &DataStream,
        data_type: &DataType,
    ) -> Result<Vec<usize>, String> {
        match (&stream.locator, data_type) {
            // CSV + TemporalSequence → [window_size, num_features]
            (DataLocatorDef::CsvDataset(def), DataType::TemporalSequence) if def.sample_mode == "temporal_window" => {
                let num_features = def.feature_columns.len();
                let window_size = def.window_size.ok_or("window_size required")?;
                Ok(vec![window_size, num_features])
            }
            // CSV + Vector → [num_features]
            (DataLocatorDef::CsvDataset(def), DataType::Vector) => {
                Ok(vec![def.feature_columns.len()])
            }
            // Image → [H, W, C] from stream.tensor_shape
            (_, DataType::Image) => {
                if stream.tensor_shape.is_empty() {
                    Err("Image stream must specify tensorShape".to_string())
                } else {
                    Ok(stream.tensor_shape.clone())
                }
            }
            _ => Err(format!("Unsupported combination: {:?} + {:?}", stream.locator, data_type))
        }
    }
    
    /// Compute output shape for Target stream
    pub fn infer_output_shape(
        stream: &DataStream,
        num_classes: usize,
    ) -> Result<Vec<usize>, String> {
        if stream.role != "Target" {
            return Err("Output shape only for Target streams".to_string());
        }
        
        match stream.data_type {
            DataType::Categorical => Ok(vec![num_classes]),
            DataType::Image => Ok(vec![stream.tensor_shape.clone()]), // Regression on images
            _ => Err(format!("Unsupported Target data type: {:?}", stream.data_type))
        }
    }
    
    /// Validate that all streams have compatible sample intersections
    pub fn validate_sample_alignment(
        &self,
        csv_loaders: &HashMap<String, CsvDatasetLoader>,
    ) -> Result<Vec<String>, String> {
        // For each stream pair:
        // - If one is temporal_window → align other to that count
        // - If both row mode → must have exact same count
        // Return valid_sample_ids
        todo!()
    }
}
```

### New Component: `DatasetValidationReport`

```typescript
export interface ValidationIssue {
    severity: 'ERROR' | 'WARNING';
    component: 'InputShape' | 'OutputShape' | 'SampleAlignment' | 'CSV' | 'Compatibility';
    message: string;
    suggestedFix?: string;
}

export interface DatasetValidationReport {
    isValid: boolean;
    issues: ValidationIssue[];
    
    // Detailed info
    inputShapes: Record<string, number[]>;   // stream_id → shape
    outputShape: number[];
    totalValidSamples: number;
    
    // Can proceed to evolution?
    canStartEvolution: boolean;
}
```

### Refactored Frontend Data Flow

```
┌─────────────────────────────────────────────────┐
│  DatasetManagerPage                             │
│  (orchestrator for dataset configuration)      │
└─────────────┬───────────────────────────────────┘
              │
              ├─→ CreateDatasetModal (CSV/Folder selection)
              │   └─→ DataStreamsPanel (stream list)
              │       └─→ DataStream component
              │           └─→ CsvDatasetConfigPanel OR GlobPatternPanel
              │
              ├─→ handleScan()
              │   ├─ Sends stream configs to scan_dataset Tauri command
              │   └─ Receives StreamScanReport[] with input_shape
              │
              └─→ handleValidate()
                  ├─ Calls Tauri validate_dataset_profile
                  ├─ Receives DatasetValidationReport
                  └─ Displays warnings/errors to user
                      with "Fix" buttons
```

### Refactored Backend Data Flow

```
┌─────────────────────────┐
│ scan_dataset command    │
└────────┬────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │ For each stream:                             │
    │ 1. Match locator (CSV/Glob/Folder/etc)      │
    │ 2. Count samples & discover classes         │
    │ 3. Infer input_shape using ShapeInference   │
    │ 4. Return StreamScanReport (with shape!)    │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ Intersection alignment:                       │
    │ - Detect temporal_window streams              │
    │ - Align all other streams to temporal count   │
    │ - Return valid_sample_ids                     │
    └────┬──────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ Return ScanDatasetResult                      │
    │ (total_matched, dropped_count, reports[])    │
    └───────────────────────────────────────────────┘

┌────────────────────────────────┐
│ validate_dataset_profile cmd  │  (NEW)
└────────┬──────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ 1. Load profile & scan results               │
    │ 2. Infer input/output shapes                 │
    │ 3. Check for issues:                         │
    │    - Empty shapes                            │
    │    - No valid samples                        │
    │    - Conflicting roles                       │
    │    - Unsupported data type combinations      │
    │ 4. Return DatasetValidationReport            │
    └──────────────────────────────────────────────┘

┌──────────────────────────────────┐
│ evaluate_population command      │
└────────┬─────────────────────────┘
         │  ← Must call validate_dataset_profile first!
         │
    ┌────▼──────────────────────────────────────────┐
    │ 1. Load cached shapes from metadata          │
    │ 2. Build Input/Output nodes with correct     │
    │    dimensions                                │
    │ 3. Train genomes                             │
    └──────────────────────────────────────────────┘
```

---

## III. DETAILED COMPONENT CHANGES

### A. Frontend: Data Stream Types & Interfaces

**File**: `src/features/dataset-manager/model/store.ts`

**Changes**:
1. Add `DataType` as discriminated union:
   ```typescript
   type DataType = 
       | { kind: 'Image'; shape: [number, number, number] }
       | { kind: 'TemporalSequence'; windowSize: number; numFeatures: number }
       | { kind: 'Vector'; numFeatures: number }
       | { kind: 'Categorical'; numClasses: number }
       | { kind: 'Text'; maxLength: number };
   ```

2. Split `DataStream` into configuration and runtime:
   ```typescript
   // Configuration (what user sets up)
   interface DataStreamConfig {
       id: string;
       alias: string;
       role: 'Input' | 'Target' | 'Ignore';
       dataTypeHint: 'Image' | 'Vector' | 'TemporalSequence' | 'Categorical' | 'Text';
       locator: DataLocatorDef;
       preprocessing?: PreprocessingSettings;
   }

   // Computed after scan
   interface DataStreamScanResult {
       configId: string;
       tensorShape: number[];
       numClasses?: number;
       sampleCount: number;
       discoveredClasses?: string[];
   }
   ```

3. Keep `DatasetProfile` but add:
   ```typescript
   interface DatasetProfile {
       // ... existing ...
       streams: DataStreamConfig[];
       scanResults?: Record<string, DataStreamScanResult>; // By stream ID
       validationReport?: DatasetValidationReport;
       isValidForEvolution: boolean;
   }
   ```

### B. Frontend: CSV Configuration Panel

**File**: `src/pages/dataset-manager-page/CsvDatasetConfigPanel.tsx`

**Changes**:
1. **Add role-aware column UI**:
   - Input stream: Show "Feature Columns" input
   - Target stream: Show "Target Column" dropdown, hide feature columns
   - This prevents user error of mixing them up

2. **Add preview that shows**:
   - First 5 rows as parsed
   - Data types of each column
   - Discovered classes (for target column)

3. **Validate on change**:
   - If role is Target and target_column is empty → warning
   - If role is Input and feature_columns are empty → warning
   - If temporal_window mode enabled but window_size not set → error

4. **Remove the hacky column range parsing** → Replace with explicit "Column 1", "Column 2" selector
   - Parse "ch0:ch11" → ["ch0", "ch1", ..., "ch11"] in backend ONLY
   - Frontend sends user-friendly config

### C. Frontend: Shape Inference & Validation UI

**File**: `src/pages/dataset-manager-page/DatasetValidationPanel.tsx` (NEW)

**Purpose**: Show user the detected shapes and any validation issues

**Display**:
```
Input Shape: [50, 12]   ←  from scan_dataset result
Output Shape: [5]       ←  from numClasses in Target stream

Issues:
⚠️  WARNING: No target column discovered
   Fix: Select "Target Column" in stream config

❌  ERROR: Output shape is empty
   Fix: Ensure Target stream has classes
```

### D. Backend: ShapeInference Service

**File**: `src-tauri/src/shape_inference.rs` (NEW)

**Methods**:
1. `infer_stream_input_shape(stream, root_path) → Result<Vec<usize>>`
   - Logic for CSV temporal: [window_size, num_features]
   - Logic for CSV vector: [num_features]
   - Logic for images: from tensor_shape

2. `infer_dataset_output_shape(target_stream, num_classes) → Result<Vec<usize>>`
   - Always returns [num_classes] for categorical
   - Error for non-Target streams

3. `validate_sample_alignment(csv_loaders, streams) → Result<Vec<String>>`
   - For each pair of streams, check sample count compatibility
   - If temporal stream exists, all others must match its sample count
   - Return final list of valid_sample_ids

### E. Backend: Enhanced scan_dataset Command

**File**: `src-tauri/src/lib.rs` → `scan_dataset`

**Changes**:
1. **For each stream, compute & return**:
   ```rust
   struct StreamScanReport {
       stream_id: String,
       alias: String,
       found_count: usize,
       missing_sample_ids: Vec<String>,
       discovered_classes: Option<Vec<String>>, // NOT HashMap
       input_shape: Option<Vec<usize>>,  // From ShapeInference
       // NEW:
       inferred_data_type: String, // "Image" | "TemporalSequence" | "Vector" | "Categorical"
       warnings: Vec<String>,      // E.g., "Temporal window count != target count"
   }
   ```

2. **Call ShapeInference for input_shape**:
   ```rust
   let shape = ShapeInference::infer_input_shape(&stream, &inferred_data_type)?;
   ```

3. **Align samples**:
   ```rust
   let valid_ids = ShapeInference::validate_sample_alignment(&csv_loaders, &streams)?;
   ```

4. **Return in ScanDatasetResult**:
   ```rust
   ScanDatasetResult {
       total_matched: valid_ids.len(),
       dropped_count: ...,
       stream_reports: vec![...],
       valid_sample_ids: valid_ids,  // NEW
   }
   ```

### F. Backend: New validate_dataset_profile Command

**File**: `src-tauri/src/lib.rs` (NEW command)

```rust
#[tauri::command]
async fn validate_dataset_profile(profile_json: String) -> Result<DatasetValidationReport, String> {
    let profile = serde_json::from_str(&profile_json)?;
    
    // 1. Check all streams have shapes
    // 2. Check Target stream exists and has numClasses
    // 3. Check sample alignment
    // 4. Check data type compatibility with layer types
    // 5. Return detailed report with issues
    
    Ok(report)
}
```

### G. Frontend: useEvolutionLoop Integration

**File**: `src/features/evolution-studio/model/useEvolutionLoop.ts`

**Changes**:
1. **Before calling startEvolution, validate**:
   ```typescript
   if (!profile.isValidForEvolution) {
       addLog("Dataset validation failed. Fix issues before proceeding.", "error");
       return;
   }
   ```

2. **Extract shapes properly**:
   ```typescript
   const inputShapes = profile.scanResults?.[inputStreamId]?.tensorShape;
   const outputShape = [profile.scanResults?.[targetStreamId]?.numClasses!];
   ```

3. **Pass to generateRandomArchitecture with data type hint**:
   ```typescript
   const randomGenome = generateRandomArchitecture(
       inputShape,
       outputShape,
       {
           maxDepth: 8,
           dataTypeHint: inputStream.dataTypeHint,  // Tell it "TemporalSequence"
       }
   );
   ```

### H. Frontend: randomArchitectureGenerator Enhancements

**File**: `src/entities/canvas-genome/lib/randomArchitectureGenerator.ts`

**Changes**:
1. **Accept dataTypeHint in options**:
   ```typescript
   interface ArchGenOptions {
       maxDepth?: number;
       useAttention?: boolean;
       dataTypeHint?: 'Image' | 'TemporalSequence' | 'Vector'; // NEW
   }
   ```

2. **Use hint to route correctly**:
   ```typescript
   if (dataTypeHint === 'TemporalSequence' && inputDimensions === 2) {
       // Always use Conv1D/LSTM/GRU
       currentNode = buildTemporalPath(nodes, currentNode, depth);
   } else if (dataTypeHint === 'Vector' && inputDimensions === 2) {
       // Use Dense layers (tabular data)
       currentNode = buildTabularPath(nodes, currentNode, depth);
   } else {
       // Infer from dimension count (image, etc.)
   }
   ```

3. **Add buildTabularPath**:
   ```typescript
   const buildTabularPath = (...): BaseNode => {
       // Dense → BatchNorm → Dropout → Dense → ...
       // No Conv layers
   }
   ```

---

## IV. IMPLEMENTATION PHASES

### Phase 1: Backend Shape Inference (2-3 weeks)

#### Tasks:
1. Create `shape_inference.rs` with core logic
2. Enhance `StreamScanReport` to include `input_shape`
3. Implement `validate_sample_alignment` for temporal windows
4. Add `validate_dataset_profile` Tauri command
5. Create comprehensive tests for:
   - CSV temporal vs row mode
   - Multi-stream alignment
   - Edge cases (empty data, missing columns)

#### Validation:
- All CSV datasets correctly report input_shape
- Temporal window alignments produce non-zero intersection
- validate_dataset_profile detects all issue categories

### Phase 2: Frontend Integration (2 weeks)

#### Tasks:
1. Update `DataStreamConfig` types
2. Enhance CSV panel with role-aware UI
3. Create `DatasetValidationPanel` component
4. Wire up shape display in preview
5. Add pre-scan validation hints

#### Validation:
- CSV panels correctly hide/show fields based on role
- Shape inference results displayed to user
- Validation errors shown with "Fix" suggestions

### Phase 3: Evolution Integration (1-2 weeks)

#### Tasks:
1. Refactor `useEvolutionLoop` to check validation before start
2. Pass dataTypeHint to random architecture generator
3. Add buildTabularPath to randomArchitectureGenerator
4. Route 2D data correctly (temporal vs tabular)

#### Validation:
- Random genomes generated with correct I/O shapes
- Conv1D used for temporal, Dense for tabular
- Output unit count matches number of classes

### Phase 4: Integration Testing & Cleanup (1 week)

#### Tasks:
1. End-to-end test: CSV temporal → scan → validate → evolve
2. End-to-end test: CSV tabular → scan → validate → evolve
3. End-to-end test: Image folder → scan → validate → evolve
4. Remove deprecated shape calculation code
5. Document dataset configuration guide

#### Validation:
- All three dataset types work in evolution
- Correct architectures generated for each type
- No shape mismatches or crashes

---

## V. DATA FLOW EXAMPLES

### Example 1: Temporal Sequence Dataset (CSV)

**User Setup**:
- CSV file: gesture.csv (19640 rows)
- Rows: timestamp | ch0 | ch1 | ... | ch11 | gesture
- Window: 50 timesteps, stride=1

**Stream Configuration**:
```
Stream 1 (Input):
  Role: Input
  DataType: TemporalSequence
  Locator: CsvDataset
    csvPath: gesture.csv
    featureColumns: ["ch0", "ch1", ..., "ch11"]  (12 features)
    sampleMode: temporal_window
    windowSize: 50

Stream 2 (Target):
  Role: Target
  DataType: Categorical
  Locator: CsvDataset
    csvPath: gesture.csv
    targetColumn: "gesture"
    sampleMode: row
```

**Scan Results**:
```
Stream 1:
  input_shape: [50, 12]  ← ShapeInference.infer_input_shape()
  sampleCount: 19591     ← 19640 - 50 + 1
  inferred_data_type: "TemporalSequence"

Stream 2:
  numClasses: 5          ← ["walk", "run", "sit", "stand", "jump"]
  sampleCount: 19640     ← Original row count (will be aligned)
  discovered_classes: ["walk", "run", "sit", "stand", "jump"]
```

**Alignment**:
```
ShapeInference.validate_sample_alignment():
  ✓ Temporal Input has 19591 samples
  ✓ Target stream has 19640 samples
  ✓ Aligning Target to temporal count: 19591 samples
  → valid_sample_ids: ["0", "1", ..., "19590"]
```

**Validation Report**:
```
Input Shape: [50, 12]
Output Shape: [5]
Total Valid Samples: 19591
Issues: None
Can Start Evolution: ✓
```

**Evolution**:
```
generateRandomArchitecture([50, 12], [5], {
  dataTypeHint: 'TemporalSequence'
}):
  → InputNode([50, 12])
  → Conv1D(32, kernel=5)
  → Conv1D(64, kernel=3)
  → Flatten()
  → Dense(128)
  → OutputNode([5])
```

---

### Example 2: Tabular Vector Dataset (CSV)

**User Setup**:
- CSV file: iris.csv (150 rows)
- Features: sepal_length | sepal_width | petal_length | petal_width | species

**Stream Configuration**:
```
Stream 1 (Input):
  Role: Input
  DataType: Vector
  Locator: CsvDataset
    csvPath: iris.csv
    featureColumns: ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    sampleMode: row

Stream 2 (Target):
  Role: Target
  DataType: Categorical
  Locator: CsvDataset
    csvPath: iris.csv
    targetColumn: "species"
    sampleMode: row
```

**Scan Results**:
```
Stream 1:
  input_shape: [4]  ← ShapeInference.infer_input_shape(): 4 features
  sampleCount: 150
  inferred_data_type: "Vector"

Stream 2:
  numClasses: 3  ← ["setosa", "versicolor", "virginica"]
  sampleCount: 150
  discovered_classes: ["setosa", "versicolor", "virginica"]
```

**Validation**:
```
Input Shape: [4]
Output Shape: [3]
Total Valid Samples: 150
Can Start Evolution: ✓
```

**Evolution**:
```
generateRandomArchitecture([4], [3], {
  dataTypeHint: 'Vector'
}):
  → InputNode([4])
  → Dense(128, relu)
  → BatchNorm()
  → Dropout(0.3)
  → Dense(64, relu)
  → Dense(32, relu)
  → OutputNode([3])
```

---

## VI. TESTING STRATEGY

### Unit Tests

**Backend** (`src-tauri/src/shape_inference.rs`):
```rust
#[test]
fn test_infer_temporal_input_shape() {
    let csv_def = CsvDatasetDef {
        sample_mode: "temporal_window",
        window_size: Some(50),
        feature_columns: vec!["ch0".to_string(), ..., "ch11".to_string()],
        ...
    };
    assert_eq!(
        ShapeInference::infer_input_shape(...),
        Ok(vec![50, 12])
    );
}

#[test]
fn test_validate_alignment_temporal_vs_row() {
    // Temporal: 19591 samples (19640 - 50 + 1)
    // Row: 19640 samples
    // Expected: Aligned to 19591
}

#[test]
fn test_infer_vector_input_shape() {
    let csv_def = CsvDatasetDef {
        sample_mode: "row",
        feature_columns: vec!["col1", "col2", "col3", "col4"],
        ...
    };
    assert_eq!(
        ShapeInference::infer_input_shape(...),
        Ok(vec![4])
    );
}
```

**Frontend** (`src/entities/canvas-genome/lib/randomArchitectureGenerator.test.ts`):
```typescript
test('TemporalSequence generates Conv1D path', () => {
    const genome = generateRandomArchitecture([50, 12], [5], {
        dataTypeHint: 'TemporalSequence'
    });
    const nodes = genome.getAllNodes();
    expect(nodes).toContainInstanceOfType(Conv1DNode);
    expect(nodes).not.toContainInstanceOfType(DenseNode);
});

test('Vector generates Dense path', () => {
    const genome = generateRandomArchitecture([4], [3], {
        dataTypeHint: 'Vector'
    });
    const nodes = genome.getAllNodes();
    expect(nodes).toContainInstanceOfType(DenseNode);
    expect(nodes).not.toContainInstanceOfType(Conv1DNode);
});
```

### Integration Tests

1. **CSV Temporal Workflow**:
   - Create dataset with temporal stream
   - Scan & validate
   - Generate random genomes
   - Verify shapes match evolution expectations

2. **CSV Vector Workflow**:
   - Create dataset with vector stream
   - Scan & validate
   - Generate random genomes
   - Verify Dense layers used

3. **Multi-Stream Alignment**:
   - CSV with temporal Input + row Target
   - Verify sample intersection is temporal count
   - Verify training only uses aligned samples

### Edge Case Tests

1. **Empty features** → Error with suggestion
2. **Missing target column** → Warning
3. **Target column not categorical** → Error
4. **Window size > row count** → Error
5. **No data rows after filtering** → Error

---

## VII. CONFIGURATION & DOCUMENTATION

### User Guide: "Configure Your Dataset"

Document the decision tree:
1. **What's your data source?**
   - Folder with images → GlobPattern
   - Single CSV file → CsvDataset
   - Multiple files with indices → MasterIndex

2. **If CSV, what's your data layout?**
   - Rows are independent samples → Row mode
   - Rows are time points within samples → Temporal window mode

3. **What are your streams?**
   - Input: Which columns have features?
   - Target: Which column has labels?

4. **Validation Checklist**:
   - [ ] All streams have valid data
   - [ ] Input/Target shapes show in validation panel
   - [ ] No warnings (or accepted warnings)
   - [ ] Sample count > 10 (recommended)

---

## VIII. ROLLBACK & SAFETY

### Breaking Changes

1. **DataStreamConfig interface changed** → All profiles need migration
   - Provide migration script to set `tensorShape` from old data
   - Mark old profiles as "needs re-scan"

2. **scan_dataset returns `input_shape`** → Frontend must handle it
   - Old frontend might ignore it → Still works but without validation benefits
   - Provide deprecation warning in return types

### Testing Before Rollout

1. Load all existing dataset profiles
2. Run scan_dataset on each
3. Verify shapes detected correctly
4. Verify old genomes still load & compile

---

## IX. FUTURE ENHANCEMENTS

### Beyond This Refactoring

1. **Multi-Input Models**:
   - Multiple Input streams (different modalities)
   - Concatenate or separate paths in random architect gen

2. **Multi-Output Models**:
   - Multiple Target streams (multi-task learning)
   - Output node generation from each target

3. **Data Augmentation Integration**:
   - CSV augmentation (mixup, noise)
   - Image augmentation (already partially supported)

4. **Streaming Data Support**:
   - Real-time CSV updates
   - Adaptive shapes based on new data

5. **Interactive Shape Visualization**:
   - Show tensor shape transformations through network
   - Highlight shape mismatches in editor

---

## X. SUCCESS CRITERIA

### After Refactoring Completion:

- [ ] All CSV datasets (row, temporal, tabular) work end-to-end
- [ ] Input shapes correctly inferred and used in architecture generation
- [ ] Output shapes correctly set to num_classes (not tensor size)
- [ ] Shape mismatches caught in validation before evolution
- [ ] Zero "CSV dataset not pre-loaded" errors in evolution
- [ ] All 3 dataset types generate appropriate architectures:
  - Temporal → Conv1D-dominant
  - Tabular → Dense-dominant
  - Images → Conv2D-dominant
- [ ] New dataset validation command available & tested
- [ ] User can understand shape errors through validation report
- [ ] All existing tests pass + 50+ new tests pass
- [ ] Code duplication for shape inference reduced >50%

---

## XI. TIMELINE ESTIMATE

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | Shape inference backend | 2-3w | Includes tests |
| 2 | Frontend integration | 2w | UI + validation panel |
| 3 | Evolution + architecture | 1-2w | Random gen routing |
| 4 | Testing & cleanup | 1w | E2E + migration |
| **Total** | | **6-9 weeks** | Assuming 40h/week |

---

## XII. RISK ASSESSMENT

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking profile format | High | Provide migration script, version profiles |
| CSV parsing edge cases | Medium | Extensive CSV test suite before rollout |
| Performance on large CSVs | Medium | Cache parsed CSVs, profile loading times |
| Backward compatibility | Medium | Accept both old & new shape formats initially |
| Shape inference bugs | High | Separate ShapeInference service with unit tests |

