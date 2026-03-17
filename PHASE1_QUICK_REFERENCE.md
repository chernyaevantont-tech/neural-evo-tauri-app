# Phase 1 Quick Reference - Backend Shape Inference

## New API Endpoints

### 1. Shape Inference Service (Rust Internal)
```rust
use crate::shape_inference::ShapeInference;

// Infer input tensor shape
let (shape, dtype, warnings) = ShapeInference::infer_input_shape(&stream, root_path)?;
// → (vec![50, 12], "TemporalSequence", vec![])

// Infer output tensor shape  
let output_shape = ShapeInference::infer_output_shape(&target_stream, num_classes)?;
// → vec![5]

// Validate sample alignment
let valid_ids = ShapeInference::validate_sample_alignment(&csv_loaders, &streams)?;
// → vec!["csv_row_0", "csv_row_1", ..., "csv_row_19590"]
```

### 2. Tauri Commands (Frontend-Callable)

#### Already Existing - Now Enhanced:
```typescript
// scan_dataset returns enhanced payload
const result = await invoke('scan_dataset', {
    rootPath: String,
    streamConfigs: StreamLocatorConfig[]
});

// result.streamReports[i] now has:
// - inferred_data_type: "TemporalSequence" | "Vector" | "Image" | "Categorical"
// - num_classes: 5  (for targets)
// - input_shape: [50, 12]  (for inputs)
// - warnings: ["warning message"]
```

#### Newly Added:
```typescript
// New validation command
const validation = await invoke('validate_dataset_profile', {
    profileJson: String  // JSON stringified DatasetProfile
});

// Returns DatasetValidationReport with:
// - isValid: boolean
// - canStartEvolution: boolean
// - issues: ValidationIssue[]
//   - severity: "ERROR" | "WARNING"
//   - component: string
//   - message: string
//   - suggestedFix: string | null
// - inputShapes: Map<streamId, shape[]>
// - outputShape: number[] | null
```

## Data Types Reference

### Data Type Strings
```
"Image"              - 2D/3D image data  [H, W, C]
"TemporalSequence"   - Time series       [T, F]  (T=timesteps, F=features)
"Vector"             - Tabular/features  [F]     (F=feature count)
"Categorical"        - Classification    [classes] (for targets)
"Text"               - Text data         (for companion files)
```

### Inferred Shapes
```
Input Stream (CSV temporal):     [50, 12]     ← window_size=50, features=12
Input Stream (CSV vector):       [4]          ← 4 features
Input Stream (Image):            [256, 256, 3]← height, width, channels
Target Stream (5 classes):       [5]          ← num_classes
Target Stream (regression):      [1]          ← single output
```

## Common Workflows

### Validate Before Evolution
```typescript
// 1. Scan dataset
const scan = await invoke('scan_dataset', {...});

// 2. Validate (with scan results)
const validation = await invoke('validate_dataset_profile', {
    profileJson: JSON.stringify(profile)
});

// 3. Check before proceeding
if (!validation.canStartEvolution) {
    validation.issues.forEach(issue => {
        if (issue.severity === "ERROR") {
            throw new Error(`${issue.message}\n\nFix: ${issue.suggestedFix}`);
        }
    });
}

// 4. Safe to start evolution
const inputShape = validation.inputShapes[inputStreamId];
const outputShape = validation.outputShape;
const numClasses = outputShape[0];
const dataTypeHint = scan.streamReports.find(r => r.streamId === inputStreamId).inferred_data_type;
```

### Route Architecture Generation
```typescript
// Use inferred_data_type for routing
const dataType = scanResult.streamReports[0].inferred_data_type;

if (dataType === "TemporalSequence") {
    // Use Conv1D/LSTM path
    architecture = buildTemporalArchitecture(inputShape, outputShape);
} else if (dataType === "Vector") {
    // Use Dense path
    architecture = buildTabularArchitecture(inputShape, outputShape);
} else if (dataType === "Image") {
    // Use Conv2D path
    architecture = buildImageArchitecture(inputShape, outputShape);
}
```

## File Locations

| Component | File |
|-----------|------|
| Core Logic | `src-tauri/src/shape_inference.rs` |
| DTOs | `src-tauri/src/dtos.rs` |
| Commands | `src-tauri/src/lib.rs` |
| Tests | `src-tauri/src/shape_inference.rs` (in module) |

## Testing

### Run All Tests
```bash
cargo test --lib
# → 13 passed (7 new + 6 existing)
```

### Run Shape Inference Tests Only
```bash
cargo test --lib shape_inference
# → 7 passed
```

### Specific Test
```bash
cargo test --lib shape_inference::tests::test_infer_temporal_input_shape
```

## Common Questions

**Q: What's the difference between `input_shape` in scan vs validation?**
A: Scan returns per-stream shape. Validation report provides the final coordinated shapes after alignment checks.

**Q: When should I use `validate_dataset_profile`?**
A: Before calling `startEvolution()`. It catches configuration issues early.

**Q: What does `inferred_data_type` control?**
A: Tells downstream code what kind of network architecture to generate (temporal→Conv1D, vector→Dense, etc.)

**Q: How are temporal samples counted?**
A: `num_samples = csv_rows - window_size + 1`
Example: 19640 rows with 50-size window → 19591 samples

**Q: What if streams have different sample counts?**
A: Alignment uses set intersection. Temporal streams determine the count for others. See `valid_sample_ids`.

## Debugging

### Check Inferred Types
```typescript
const scan = await invoke('scan_dataset', {...});
scan.streamReports.forEach(report => {
    console.log(`${report.alias}: ${report.inferred_data_type}`);
    if (report.warnings.length > 0) {
        console.warn(`⚠️  Warnings:`, report.warnings);
    }
});
```

### Check Validation Issues  
```typescript
const validation = await invoke('validate_dataset_profile', {...});
validation.issues.forEach(issue => {
    console.error(`[${issue.severity}] ${issue.component}`);
    console.error(`  Message: ${issue.message}`);
    console.error(`  Fix: ${issue.suggestedFix}`);
});
```

## Performance Notes

- ✅ Shape inference is O(1) - doesn't process data
- ✅ Sample alignment is O(n_streams) - very fast
- ✅ Validation is O(n_streams) - runs in ms
- ⚠️ CSV scan does full file read - time depends on file size

## Error Handling

All functions return `Result<T, String>`:
```rust
match ShapeInference::infer_input_shape(&stream, root) {
    Ok((shape, dtype, warnings)) => {
        // Use shape and dtype
        if !warnings.is_empty() {
            // Log warnings
        }
    }
    Err(e) => {
        // Handle error
    }
}
```

## Future Enhancements

These are planned for later phases:
1. Compatibility checks between layer types and data types
2. Caching of CSV loaders for performance
3. Stratified sample validation
4. Multi-input/output model support
5. Automatic architecture recommendations based on data type

---

**Last Updated**: 2026-03-17
**Status**: Phase 1 Complete ✅
