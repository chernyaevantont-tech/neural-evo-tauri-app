# CSV Dataset Flexible System

## 📋 Vision

**Одна унифицированная система `CsvDatasetDef`** которая позволяет пользователю определить:
1. **Что такое "образец"** в CSV? (строка vs. скользящее окно)
2. **Какие столбцы = features?** (multi-select с range parsing)
3. **Какой столбец = target?**
4. **Как нормализовать?** (per-channel, per-sample, global)

---

## 🎯 Use Cases

### Case 1: Жесты (ваш датасет) — TEMPORAL
```
Data: ticks_gestures.csv (1000 строк, 12 каналов)

Config:
{
  sampleMode: "temporal_window",
  windowSize: 50,
  windowStride: 1,
  featureColumns: "ch0:ch11",
  targetColumn: "gesture"
}

Result: 951 samples × [50, 12] tensor
```

### Case 2: Iris Dataset — TABULAR
```
Data: iris.csv (150 строк, 4 признака)

Config:
{
  sampleMode: "row",
  featureColumns: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  targetColumn: "species"
}

Result: 150 samples × [4] вектор
```

### Case 3: Sensor Data — HYBRID
```
Data: sensors.csv (10000 строк, 8 датчиков)

Config (вариант 1 - Row):
{
  sampleMode: "row",
  featureColumns: "sensor_1:sensor_8",
  targetColumn: "failure_class"
}
Result: 10000 samples × [8] вектор

Config (вариант 2 - Temporal):
{
  sampleMode: "temporal_window",
  windowSize: 100,
  windowStride: 10,
  featureColumns: "sensor_1:sensor_8",
  targetColumn: "failure_class"
}
Result: ~991 samples × [100, 8] тензор
```

---

## 🏗️ Architecture

### **Type Definitions (dtos.rs + TypeScript)**

```typescript
// === TypeScript ===
export type SampleMode = 'row' | 'temporal_window';

export interface CsvDatasetDef {
  // CSV location
  csvPath: string;
  hasHeaders: boolean;
  
  // How to define a sample
  sampleMode: SampleMode;
  
  // Columns
  featureColumns: string[] | { range: string };  // "ch0:ch11" or array
  targetColumn: string;
  multiTargetColumns?: string[];  // For multi-label
  
  // Temporal params (only if sampleMode === 'temporal_window')
  windowSize?: number;
  windowStride?: number;
  
  // Preprocessing
  preprocessing: {
    normalization: 'none' | 'per-channel' | 'per-sample' | 'global';
    handleMissing: 'skip' | 'interpolate' | 'mean';
  };
}

// === Rust ===
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataLocatorDef {
    GlobPattern(GlobPatternDef),
    FolderMapping,
    CompanionFile(CompanionFileDef),
    MasterIndex(MasterIndexDef),
    CsvDataset(CsvDatasetDef),  // ← NEW
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvDatasetDef {
    pub csv_path: String,
    pub has_headers: bool,
    pub sample_mode: String,  // "row" | "temporal_window"
    pub feature_columns: Vec<String>,  // Pre-parsed
    pub target_column: String,
    pub window_size: Option<usize>,
    pub window_stride: Option<usize>,
    pub preprocessing: PreprocessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub normalization: String,  // "none" | "per-channel" | etc
    pub handle_missing: String,
}
```

---

## 🖥️ Frontend UI

### **CsvDatasetConfigPanel** (New Component)

```
┌──────────────────────────────────────────────────────┐
│  CSV Dataset Configuration                           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  CSV File Path:  [ticks_gestures.csv] [Browse...]   │
│  □ Has Headers                                      │
│                                                      │
├──────────────────────────────────────────────────────┤
│ SAMPLE MODE (How to group rows into samples)        │
│                                                      │
│  ◉ Row Mode      (Each row = one sample)           │
│  ○ Temporal Window (Sliding window = one sample)   │
│                                                      │
│  [If Temporal Window selected:]                     │
│  Window Size:    [50] ← ────────┐                  │
│  Window Stride:  [1]  ← ────────┘ Overlapping     │
│                                                      │
├──────────────────────────────────────────────────────┤
│ FEATURE COLUMNS (Which columns = input data)        │
│                                                      │
│  Method:  ○ Range [ch0 : ch11] ✓                  │
│           ○ Manual Selection                        │
│                                                      │
│  Selected: ch0, ch1, ..., ch11 (12 columns)        │
│  Shape inference: (depends on mode)                │
│    ROW MODE → [12]                                 │  
│    TEMPORAL → [50, 12]                             │
│                                                      │
│  [Preview Selected Columns]                         │
│                                                      │
├──────────────────────────────────────────────────────┤
│ TARGET COLUMN (Label/Class)                         │
│                                                      │
│  Select:  [gesture ▼]                              │
│           Unique values: 5 (wave, rotate, ...)     │
│                                                      │
│  ○ Single target (classification)                  │
│  ○ Multi-target (multi-label)                      │
│    [Add target column...]                          │
│                                                      │
├──────────────────────────────────────────────────────┤
│ PREPROCESSING                                       │
│                                                      │
│  Normalization:  [per-channel ▼]                   │
│    ○ None        - raw values                      │
│    ○ Global      - single mean/std                 │
│    ○ Per-sample  - (T, C) → norm within sample    │
│    ◉ Per-channel - each channel independently      │
│                                                      │
│  Missing Values: [skip ▼]                          │
│    ○ Skip        - discard rows with NaN          │
│    ○ Interpolate - linear fill                     │
│    ○ Mean        - column mean fill                │
│                                                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  [Preview] [Save Config]                            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### **CsvPreviewModal** (Enhanced)

```
┌─────────────────────────────────────────────┐
│ CSV Preview                                 │
├─────────────────────────────────────────────┤
│                                             │
│ Raw CSV:                                    │
│ ┌─────────────────────────────────────────┐│
│ │ gesture | ch0   | ch1   | ... | ch11    ││
│ │ gesture | ch0   | ch1   | ... | ch11    ││
│ │ gesture | 6660  | 6232  | ... | 5856    ││
│ │ gesture | 6612  | 6190  | ... | 5858    ││
│ │ ...                                     ││
│ └─────────────────────────────────────────┘│
│                                             │
│ Sample Creation Preview:                    │
│ ┌─────────────────────────────────────────┐│
│ │ Mode: temporal_window                   ││
│ │ Window: 50 timesteps, Stride: 1         ││
│ │ Result: 951 samples from 1000 rows      ││
│ │                                         ││
│ │ SAMPLE 0 (rows 0-49):                   ││
│ │ Ch0: [6660, 6612, 6570, ...]           ││
│ │ Ch1: [6232, 6190, 6156, ...]           ││
│ │ ...                                     ││
│ │ Ch11: [5856, 5858, 5862, ...]          ││
│ │ Label: wave                             ││
│ │                                         ││
│ │ SAMPLE 1 (rows 1-50):                   ││
│ │ Ch0: [6612, 6570, ..., ]               ││
│ │ ...                                     ││
│ └─────────────────────────────────────────┘│
│                                             │
└─────────────────────────────────────────────┘
```

---

## ⚙️ Backend Implementation

### **data_loader.rs — CsvDatasetLoader**

```rust
pub struct CsvDatasetLoader {
    csv_path: PathBuf,
    config: CsvDatasetDef,
    
    // Row-based cache
    rows: Vec<Vec<f32>>,  // [num_rows, num_features]
    labels: Vec<String>,  // [num_rows]
    
    // Metadata
    num_samples: usize,
    feature_indices: Vec<usize>,
    target_index: usize,
}

impl CsvDatasetLoader {
    /// Load entire CSV into memory
    pub fn init(root: &Path, config: &CsvDatasetDef) -> Result<Self> {
        let csv_path = root.join(&config.csv_path);
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.has_headers)
            .from_path(&csv_path)?;
        
        // Parse feature_columns range (if "ch0:ch11")
        let feature_indices = parse_column_range(&config.feature_columns, &reader)?;
        let target_index = find_column(&config.target_column, &reader)?;
        
        // Load all rows
        let mut rows = vec![];
        let mut labels = vec![];
        
        for record in reader.records() {
            let record = record?;
            
            // Extract features
            let values: Vec<f32> = feature_indices.iter()
                .map(|&i| record.get(i).unwrap_or("").parse().ok())
                .collect::<Option<Vec<_>>>()?;
            rows.push(values);
            
            // Extract label
            let label = record.get(target_index).unwrap_or("").to_string();
            labels.push(label);
        }
        
        // Compute number of samples
        let num_samples = match config.sample_mode.as_str() {
            "row" => rows.len(),
            "temporal_window" => {
                let ws = config.window_size.unwrap_or(50);
                rows.len().saturating_sub(ws - 1)
            }
            _ => return Err("Unknown sample_mode".into()),
        };
        
        Ok(CsvDatasetLoader {
            csv_path,
            config: config.clone(),
            rows,
            labels,
            num_samples,
            feature_indices,
            target_index,
        })
    }
    
    /// Load a single sample by ID
    pub fn load_sample(&self, sample_id: &str, device: &Device) 
        -> Result<SampleData> 
    {
        let idx = sample_id.parse::<usize>()?;
        
        match self.config.sample_mode.as_str() {
            "row" => {
                // Each row = one sample
                let features = &self.rows[idx];
                let label = &self.labels[idx];
                
                // Shape depends on num features
                let tensor = Tensor::from_floats(features.clone())
                    .reshape([features.len()]);
                
                Ok(SampleData {
                    input: tensor,
                    label: label.clone(),
                })
            }
            "temporal_window" => {
                // Sliding window
                let ws = self.config.window_size.unwrap_or(50);
                let start_row = idx;
                let end_row = idx + ws;
                
                if end_row > self.rows.len() {
                    return Err("Invalid window range".into());
                }
                
                // Stack rows[start:end] → [T, C] tensor
                let mut window = vec![];
                for row_idx in start_row..end_row {
                    window.push(self.rows[row_idx].clone());
                }
                
                // Apply normalization
                window = self.normalize_temporal(&window)?;
                
                // Shape: [ws, num_features]
                let flat: Vec<f32> = window.iter()
                    .flat_map(|r| r.iter())
                    .cloned()
                    .collect();
                let tensor = Tensor::from_floats(flat)
                    .reshape([ws, self.feature_indices.len()]);
                
                // Get label from first row (or consensus?)
                let label = self.labels[start_row].clone();
                
                Ok(SampleData {
                    input: tensor,
                    label,
                })
            }
            _ => Err("Unknown sample_mode".into()),
        }
    }
    
    /// Normalize temporal window [T, C]
    fn normalize_temporal(&self, window: &Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        match self.config.preprocessing.normalization.as_str() {
            "none" => Ok(window.clone()),
            
            "per-channel" => {
                // For each channel: z-score normalize
                let num_channels = window[0].len();
                let mut normalized = window.clone();
                
                for ch in 0..num_channels {
                    let vals: Vec<f32> = window.iter()
                        .map(|row| row[ch])
                        .collect();
                    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                    let var = vals.iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f32>() / vals.len() as f32;
                    let std = var.sqrt();
                    
                    for t in 0..window.len() {
                        if std > 1e-7 {
                            normalized[t][ch] = (window[t][ch] - mean) / std;
                        }
                    }
                }
                Ok(normalized)
            }
            
            "per-sample" => {
                // Normalize entire [T, C] window together
                let all_vals: Vec<f32> = window.iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect();
                let mean = all_vals.iter().sum::<f32>() / all_vals.len() as f32;
                let var = all_vals.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f32>() / all_vals.len() as f32;
                let std = var.sqrt();
                
                let normalized = window.iter()
                    .map(|row| {
                        row.iter()
                            .map(|v| if std > 1e-7 { (*v - mean) / std } else { 0.0 })
                            .collect()
                    })
                    .collect();
                Ok(normalized)
            }
            
            _ => Err("Unknown normalization".into()),
        }
    }
}
```

### **lib.rs — Scanning**

```rust
pub fn scan_csv_dataset(
    root: &Path,
    config: &CsvDatasetDef,
) -> Result<ScanResult> {
    let loader = CsvDatasetLoader::init(root, config)?;
    
    // Count samples
    let total_count = loader.num_samples;
    
    // Introspect labels
    let unique_labels: HashSet<_> = loader.labels.iter().cloned().collect();
    
    Ok(ScanResult {
        total_matched: total_count,
        discovered_classes: unique_labels.into_iter().collect(),
        stream_reports: vec![
            format!("CSV loaded: {} total rows", loader.rows.len()),
            format!("Sample mode: {}", config.sample_mode),
            if config.sample_mode == "temporal_window" {
                format!("  Window: {} timesteps, stride: {}",
                    config.window_size.unwrap_or(50),
                    config.window_stride.unwrap_or(1))
            } else {
                "".to_string()
            },
            format!("Features: {} channels", loader.feature_indices.len()),
            format!("Samples: {}", total_count),
        ],
    })
}
```

---

## 🎨 UI Components (React)

### **CsvDatasetConfigPanel**

```typescript
// File: src/pages/dataset-manager-page/CsvDatasetConfigPanel.tsx

import { useState } from 'react';
import { CsvDatasetDef } from '@/shared/lib';

export const CsvDatasetConfigPanel = ({ onSave }: { onSave: (config: CsvDatasetDef) => void }) => {
  const [config, setConfig] = useState<CsvDatasetDef>({
    csvPath: '',
    hasHeaders: true,
    sampleMode: 'row',
    featureColumns: [],
    targetColumn: '',
    preprocessing: {
      normalization: 'per-channel',
      handleMissing: 'skip',
    },
  });

  const [csvPreview, setCsvPreview] = useState<string[][] | null>(null);

  const handleCsvPathChange = async (path: string) => {
    setConfig(prev => ({ ...prev, csvPath: path }));
    // Load CSV preview
    const preview = await invoke<string[][]>('preview_csv', { path, rows: 5 });
    setCsvPreview(preview);
  };

  const handleFeatureColumnsChange = (columns: string[] | string) => {
    if (typeof columns === 'string') {
      // Parse range "ch0:ch11"
      const parsed = parseColumnRange(columns);
      setConfig(prev => ({ ...prev, featureColumns: parsed }));
    } else {
      setConfig(prev => ({ ...prev, featureColumns: columns }));
    }
  };

  const toggleTemporalMode = () => {
    setConfig(prev => ({
      ...prev,
      sampleMode: prev.sampleMode === 'row' ? 'temporal_window' : 'row',
    }));
  };

  return (
    <div className={styles.panel}>
      {/* CSV Path */}
      <div className={styles.section}>
        <label>CSV File Path</label>
        <input
          type="text"
          value={config.csvPath}
          onChange={e => handleCsvPathChange(e.target.value)}
          placeholder="data.csv"
        />
        <label>
          <input
            type="checkbox"
            checked={config.hasHeaders}
            onChange={e => setConfig(prev => ({ ...prev, hasHeaders: e.target.checked }))}
          />
          Has Headers
        </label>
      </div>

      {/* Sample Mode */}
      <div className={styles.section}>
        <label>Sample Mode</label>
        <label>
          <input
            type="radio"
            name="sampleMode"
            checked={config.sampleMode === 'row'}
            onChange={toggleTemporalMode}
          />
          Row Mode (each row = 1 sample)
        </label>
        <label>
          <input
            type="radio"
            name="sampleMode"
            checked={config.sampleMode === 'temporal_window'}
            onChange={toggleTemporalMode}
          />
          Temporal Window (sliding window = 1 sample)
        </label>

        {config.sampleMode === 'temporal_window' && (
          <div className={styles.temporal}>
            <label>
              Window Size:
              <input
                type="number"
                value={config.windowSize || 50}
                onChange={e => setConfig(prev => ({ ...prev, windowSize: parseInt(e.target.value) }))}
                min={1}
              />
            </label>
            <label>
              Window Stride:
              <input
                type="number"
                value={config.windowStride || 1}
                onChange={e => setConfig(prev => ({ ...prev, windowStride: parseInt(e.target.value) }))}
                min={1}
              />
            </label>
          </div>
        )}
      </div>

      {/* Feature Columns */}
      <div className={styles.section}>
        <label>Feature Columns</label>
        <input
          type="text"
          value={typeof config.featureColumns === 'string' ? config.featureColumns : config.featureColumns.join(', ')}
          onChange={e => handleFeatureColumnsChange(e.target.value)}
          placeholder="ch0:ch11 or col1, col2, col3"
        />
        <small>Use range (ch0:ch11) or comma-separated list</small>
      </div>

      {/* Target Column */}
      <div className={styles.section}>
        <label>Target Column (Label)</label>
        <input
          type="text"
          value={config.targetColumn}
          onChange={e => setConfig(prev => ({ ...prev, targetColumn: e.target.value }))}
          placeholder="gesture"
        />
      </div>

      {/* Preprocessing */}
      <div className={styles.section}>
        <label>Normalization</label>
        <select
          value={config.preprocessing.normalization}
          onChange={e => setConfig(prev => ({
            ...prev,
            preprocessing: { ...prev.preprocessing, normalization: e.target.value as any }
          }))}
        >
          <option value="none">None</option>
          <option value="global">Global</option>
          <option value="per-sample">Per-Sample</option>
          <option value="per-channel">Per-Channel</option>
        </select>

        <label>Missing Values</label>
        <select
          value={config.preprocessing.handleMissing}
          onChange={e => setConfig(prev => ({
            ...prev,
            preprocessing: { ...prev.preprocessing, handleMissing: e.target.value as any }
          }))}
        >
          <option value="skip">Skip</option>
          <option value="interpolate">Interpolate</option>
          <option value="mean">Mean Fill</option>
        </select>
      </div>

      {/* Preview */}
      {csvPreview && (
        <div className={styles.preview}>
          <h4>CSV Preview</h4>
          <table>
            <tbody>
              {csvPreview.slice(0, 5).map((row, i) => (
                <tr key={i}>
                  {row.map((val, j) => (
                    <td key={j}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <button onClick={() => onSave(config)}>Save Configuration</button>
    </div>
  );
};

function parseColumnRange(input: string): string[] {
  // Parse "ch0:ch11" → ["ch0", "ch1", ..., "ch11"]
  const match = input.match(/(\w+)(\d+):(\w+)(\d+)/);
  if (match) {
    const [, prefix, start, , end] = match;
    const result = [];
    for (let i = parseInt(start); i <= parseInt(end); i++) {
      result.push(`${prefix}${i}`);
    }
    return result;
  }
  // Fallback: comma-separated
  return input.split(',').map(s => s.trim());
}
```

---

## 📊 Sample Mode Comparison

| Aspect | Row Mode | Temporal Window |
|--------|----------|-----------------|
| **Definition** | Each CSV row = 1 sample | N consecutive rows = 1 sample |
| **Sample ID** | Row index | Window start index |
| **Input Shape** | [num_features] | [window_size, num_features] |
| **Use Case** | Tabular, independent rows | Time series, sequential patterns |
| **Example Datasets** | Iris, diabetes, credit | Gesture recognition, sensor data |
| **Overlapping** | No | Yes (stride controls overlap) |
| **Data Amount** | csv_rows samples | (csv_rows - window_size + 1) samples |

---

## 🔄 Data Flow

```
User Config:
├─ csvPath: "ticks_gestures.csv"
├─ sampleMode: "temporal_window"
├─ featureColumns: "ch0:ch11"
├─ targetColumn: "gesture"
├─ windowSize: 50
├─ windowStride: 1
└─ preprocessing: per-channel

   ↓ Frontend parses → CsvDatasetDef

   ↓ Tauri IPC → Backend

CsvDatasetLoader::init()
├─ Load CSV (1000 rows × 12 cols)
├─ Parse config
├─ Validate columns
└─ Compute num_samples = 1000 - 50 + 1 = 951

   ↓ Training/Evolution loop

load_sample("0"):
├─ Rows [0:50] from CSV
├─ Extract ch0:ch11 → [50, 12] tensor
├─ Normalize per-channel
└─ Return (input=[50, 12], label="wave")

load_sample("1"):
├─ Rows [1:51] from CSV
├─ ... (overlapping window)
```

---

## ✅ Implementation Checklist

### Phase 1: Core Types & Backend (Priority 0)
- [ ] Add `CsvDatasetDef` to `dtos.rs`
- [ ] Add `TemporalSequence` to `DataType` enum
- [ ] Implement `CsvDatasetLoader` in `data_loader.rs`
- [ ] Implement `scan_csv_dataset()` in `lib.rs`
- [ ] Update `load_sample()` to handle `CsvDataset` locator
- [ ] Implement column range parsing ("ch0:ch11")
- [ ] Implement per-channel normalization

### Phase 2: Frontend (Priority 0)
- [ ] Create `CsvDatasetConfigPanel` component
- [ ] Integrate into `DataStreamsPanel`
- [ ] Create enhanced `CsvPreviewModal`
- [ ] Update `ProfileState` to support CsvDataset

### Phase 3: Integration (Priority 1)
- [ ] Update `scan_dataset` Tauri command
- [ ] Update `evaluate_population` to use new loader
- [ ] Stratified split for CSV-based data
- [ ] End-to-end test with gesture dataset

### Phase 4: Polish (Priority 2)
- [ ] Multi-target support
- [ ] Additional preprocessing options
- [ ] Performance optimization (csv caching)
- [ ] Error handling & validation

---

## 🎓 Example: Using Your Dataset

**Your CSV**: `ticks_gestures.csv`
```
gesture,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11
wave,6660,6232,5850,6820,5558,5450,7796,7036,5552,5742,5782,5856
wave,6612,6190,5842,6746,5560,5448,7786,7038,5562,5750,5786,5858
rotate,6570,6156,5846,6700,5554,5452,7790,6976,5566,5752,5786,5862
rotate,6522,6104,5820,6650,5554,5446,7780,6938,5560,5740,5786,5852
...
```

**UI Configuration:**
```
CSV Path: ticks_gestures.csv
Has Headers: ☑
Sample Mode: ○ Row   ◉ Temporal Window
  Window Size: 50
  Window Stride: 1
Feature Columns: ch0:ch11
Target Column: gesture
Normalization: Per-Channel
Missing Values: Skip
[Save]
```

**Result:**
- 1000 CSV rows
- 951 overlapping temporal samples
- Each sample: [50 timesteps, 12 channels]
- Classes: {wave, rotate, ...}
- Ready for Conv1D → Dense network

---

## 🚀 Next Steps

1. Review this architecture
2. Decide if this matches your vision
3. Begin Phase 1 implementation (types + backend)
4. Iterate with your dataset

