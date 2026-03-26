use burn::tensor::Tensor;
use std::path::{Path, PathBuf};

use crate::dtos::CsvDatasetDef;
use crate::entities::DynamicTensor;

type Backend = crate::backend::TrainBackend;
type TrainDevice = crate::backend::TrainDevice;

/// Loads CSV-based datasets in both row-wise and temporal-window modes
#[allow(dead_code)]
#[derive(Clone)]
pub struct CsvDatasetLoader {
    csv_path: PathBuf,
    config: CsvDatasetDef,
    
    // All raw data loaded into memory
    rows: Vec<Vec<f32>>,           // [num_rows, num_features]
    labels: Vec<String>,           // [num_rows]
    feature_indices: Vec<usize>,   // Which CSV columns are features
    target_index: usize,           // Which CSV column is the target
    
    // Computed metadata
    pub num_samples: usize,
    pub discovered_classes: Vec<String>,
}

impl CsvDatasetLoader {
    /// Initialize the CSV loader by reading the entire CSV into memory
    pub fn init(root: &Path, config: CsvDatasetDef) -> Result<Self, String> {
        let csv_path = root.join(&config.csv_path);
        
        if !csv_path.exists() {
            return Err(format!("CSV file not found: {}", csv_path.display()));
        }

        // Open CSV reader first to get headers
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.has_headers)
            .from_path(&csv_path)
            .map_err(|e| format!("Failed to read CSV: {}", e))?;

        // Get headers first
        let headers = if config.has_headers {
            reader.headers()
                .map_err(|e| format!("Failed to read headers: {}", e))?
                .iter()
                .map(|h| h.to_string())
                .collect::<Vec<_>>()
        } else {
            // If no headers, generate numeric indices
            (0..100).map(|i| i.to_string()).collect()
        };

        // Now map feature column names/specs to actual column indices
        let feature_indices = if config.feature_columns.is_empty() {
            vec![] // Target streams have no features
        } else {
            map_column_specs_to_indices(&config.feature_columns, &headers)?
        };

        // Find target column index (required if target_column is not empty, optional otherwise)
        let target_index = if config.target_column.is_empty() {
            // Input stream: no target column needed
            0 // dummy value, won't be used
        } else if config.has_headers {
            headers.iter()
                .position(|h| h == &config.target_column)
                .ok_or_else(|| format!("Target column '{}' not found in headers: {:?}", config.target_column, headers))?
        } else {
            config.target_column
                .parse::<usize>()
                .map_err(|_| format!("Invalid column index for target_column '{}': expected usize", config.target_column))?
        };

        // Validate feature indices are within bounds
        if let Some(&max_idx) = feature_indices.iter().max() {
            if max_idx >= headers.len() {
                return Err(format!(
                    "Feature column index {} exceeds CSV width {}",
                    max_idx,
                    headers.len()
                ));
            }
        }

        // Load all data from CSV
        let mut rows = vec![];
        let mut labels = vec![];
        let mut class_set = std::collections::HashSet::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record = result.map_err(|e| format!("Error reading row {}: {}", row_idx, e))?;

            // Extract feature values
            let mut feature_values = vec![];
            for &col_idx in &feature_indices {
                let val_str = record.get(col_idx).unwrap_or("0");
                let val = val_str.trim().parse::<f32>()
                    .map_err(|_| format!(
                        "Failed to parse float at row {} col {}: '{}'",
                        row_idx, col_idx, val_str
                    ))?;
                feature_values.push(val);
            }
            rows.push(feature_values);

            // Extract label (required if target_column is specified, dummy "unlabeled" for Input streams)
            let label = if config.target_column.is_empty() {
                // Input stream: use dummy label
                "unlabeled".to_string()
            } else {
                // Target stream: extract from target column
                record.get(target_index)
                    .ok_or_else(|| format!("Missing target value at row {}", row_idx))?
                    .to_string()
            };
            labels.push(label.clone());
            class_set.insert(label);
        }

        if rows.is_empty() {
            return Err("CSV contains no data rows".to_string());
        }

        // Compute number of samples based on sample mode
        let num_samples = match config.sample_mode.as_str() {
            "row" => rows.len(),
            "temporal_window" => {
                let ws = config.window_size.ok_or("window_size required for temporal_window mode")?;
                if rows.len() < ws {
                    return Err(format!(
                        "CSV has {} rows but window_size is {}",
                        rows.len(),
                        ws
                    ));
                }
                rows.len().saturating_sub(ws - 1)
            }
            _ => return Err(format!("Unknown sample_mode: {}", config.sample_mode)),
        };

        let mut discovered_classes: Vec<String> = class_set.into_iter().collect();
        discovered_classes.sort();

        Ok(CsvDatasetLoader {
            csv_path,
            config,
            rows,
            labels,
            feature_indices,
            target_index,
            num_samples,
            discovered_classes,
        })
    }

    /// Retrieve raw label string for stratification
    pub fn get_label(&self, sample_idx: usize) -> Option<String> {
        if sample_idx < self.labels.len() {
            Some(self.labels[sample_idx].clone())
        } else {
            None
        }
    }

    /// Load a single sample by index
    pub fn load_sample(
        &self,
        sample_idx: usize,
        device: &TrainDevice,
    ) -> Result<(DynamicTensor<Backend>, String), String> {
        match self.config.sample_mode.as_str() {
            "row" => {
                // Each row = one sample
                if sample_idx >= self.rows.len() {
                    return Err(format!("Sample index {} out of bounds", sample_idx));
                }

                let features = &self.rows[sample_idx];
                let label = &self.labels[sample_idx];

                // If no features (e.g., Target stream), return a dummy tensor
                let tensor_2d = if self.feature_indices.is_empty() {
                    let tensor = Tensor::<Backend, 2>::from_data([[0.0]], device);
                    tensor // Dummy tensor
                } else {
                    // Normalize if configured
                    let features = self.normalize_row(features)?;

                    // Create tensor [1, num_features]
                    let tensor = Tensor::<Backend, 1>::from_floats(features.as_slice(), device);
                    tensor.reshape([1, features.len()])
                };

                Ok((DynamicTensor::Dim2(tensor_2d), label.clone()))
            }

            "temporal_window" => {
                let ws = self.config.window_size
                    .ok_or("window_size required for temporal_window")?;
                
                if sample_idx >= self.num_samples {
                    return Err(format!(
                        "Sample index {} out of bounds for {} samples",
                        sample_idx, self.num_samples
                    ));
                }

                let start_row = sample_idx;
                let end_row = sample_idx + ws;

                // Stack rows[start:end] into [T, C] format
                let mut window_data = vec![];
                for row_idx in start_row..end_row {
                    window_data.extend(&self.rows[row_idx]);
                }

                // If no features (e.g., Target stream), return a dummy 3D tensor
                let tensor_3d = if self.feature_indices.is_empty() {
                    let tensor = Tensor::<Backend, 3>::from_data([[[0.0]]], device);
                    tensor // Dummy tensor
                } else {
                    // Apply normalization
                    let window_data = self.normalize_temporal(&window_data, ws)?;

                    // Create tensor [1, ws, num_features]
                    let tensor = Tensor::<Backend, 1>::from_floats(window_data.as_slice(), device);
                    tensor.reshape([1, ws, self.feature_indices.len()])
                };

                // Use label from first row of window
                let label = self.labels[start_row].clone();

                Ok((DynamicTensor::Dim3(tensor_3d), label))
            }

            _ => Err(format!("Unknown sample_mode: {}", self.config.sample_mode)),
        }
    }

    /// Normalize a single row
    fn normalize_row(&self, row: &[f32]) -> Result<Vec<f32>, String> {
        match self.config.preprocessing.normalization.as_str() {
            "none" => Ok(row.to_vec()),
            
            "global" => {
                // Not applicable for single row
                Ok(row.to_vec())
            }
            
            "per-sample" => {
                // Normalize single row to mean 0, std 1
                let mean = row.iter().sum::<f32>() / row.len() as f32;
                let var = row.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f32>() / row.len() as f32;
                let std = var.sqrt();

                let normalized = if std > 1e-7 {
                    row.iter().map(|v| (v - mean) / std).collect()
                } else {
                    row.iter().map(|_| 0.0).collect()
                };
                Ok(normalized)
            }

            _ => Err(format!(
                "Unknown normalization for row mode: {}",
                self.config.preprocessing.normalization
            )),
        }
    }

    /// Normalize temporal window [T * C] flattened
    fn normalize_temporal(
        &self,
        window: &[f32],
        window_size: usize,
    ) -> Result<Vec<f32>, String> {
        let num_channels = self.feature_indices.len();
        
        if window.len() != window_size * num_channels {
            return Err(format!(
                "Window size mismatch: expected {}, got {}",
                window_size * num_channels,
                window.len()
            ));
        }

        match self.config.preprocessing.normalization.as_str() {
            "none" => Ok(window.to_vec()),

            "per-channel" => {
                let mut normalized = window.to_vec();
                
                // For each channel independently
                for ch in 0..num_channels {
                    let mut channel_vals = vec![];
                    for t in 0..window_size {
                        channel_vals.push(window[t * num_channels + ch]);
                    }

                    let mean = channel_vals.iter().sum::<f32>() / window_size as f32;
                    let var = channel_vals.iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f32>() / window_size as f32;
                    let std = var.sqrt();

                    // Normalize this channel
                    for t in 0..window_size {
                        let idx = t * num_channels + ch;
                        normalized[idx] = if std > 1e-7 {
                            (window[idx] - mean) / std
                        } else {
                            0.0
                        };
                    }
                }

                Ok(normalized)
            }

            "per-sample" => {
                // Normalize entire window together
                let mean = window.iter().sum::<f32>() / window.len() as f32;
                let var = window.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f32>() / window.len() as f32;
                let std = var.sqrt();

                let normalized = window.iter()
                    .map(|v| if std > 1e-7 { (*v - mean) / std } else { 0.0 })
                    .collect();
                Ok(normalized)
            }

            _ => Err(format!(
                "Unknown normalization: {}",
                self.config.preprocessing.normalization
            )),
        }
    }

    pub fn get_num_features(&self) -> usize {
        self.feature_indices.len()
    }

    pub fn get_window_size(&self) -> Option<usize> {
        self.config.window_size
    }
}

/// Map column specifications to actual indices using CSV headers
/// Supports: "ch0" (column name), "0" (direct index), "ch0:ch11" (range by name)
fn map_column_specs_to_indices(specs: &[String], headers: &[String]) -> Result<Vec<usize>, String> {
    if specs.is_empty() {
        return Ok(vec![]);
    }

    let mut indices = vec![];
    
    for spec in specs {
        let spec = spec.trim();
        
        // Try range format: "ch0:ch11" or "0:11"
        if let Some(colon_pos) = spec.find(':') {
            let start_spec = spec[..colon_pos].trim();
            let end_spec = spec[colon_pos + 1..].trim();
            
            let start_idx = resolve_single_column(start_spec, headers)?;
            let end_idx = resolve_single_column(end_spec, headers)?;
            
            if start_idx <= end_idx {
                for i in start_idx..=end_idx {
                    indices.push(i);
                }
            } else {
                return Err(format!("Invalid column range: {}:{} (start > end)", start_spec, end_spec));
            }
        } else {
            // Single column (by name or index)
            let idx = resolve_single_column(spec, headers)?;
            indices.push(idx);
        }
    }

    Ok(indices)
}

/// Resolve a single column spec to its index
/// Tries: direct index → column name lookup
fn resolve_single_column(spec: &str, headers: &[String]) -> Result<usize, String> {
    let spec = spec.trim();
    
    // Try parsing as direct index first
    if let Ok(idx) = spec.parse::<usize>() {
        if idx < headers.len() {
            return Ok(idx);
        } else {
            return Err(format!("Column index {} out of bounds (CSV has {} columns)", idx, headers.len()));
        }
    }
    
    // Try looking up by column name
    if let Some(pos) = headers.iter().position(|h| h == spec) {
        return Ok(pos);
    }
    
    // Try extracting number from column name like "ch0"
    if let Ok(num) = extract_trailing_number(spec) {
        // Look for a column that has this number as suffix
        if let Some(pos) = headers.iter().position(|h| {
            if let Ok(h_num) = extract_trailing_number(h) {
                h_num == num
            } else {
                false
            }
        }) {
            return Ok(pos);
        }
    }
    
    Err(format!(
        "Cannot resolve column spec '{}'. Available columns: {:?}",
        spec, headers
    ))
}

/// Parse feature column range like "ch0:ch11" or use predefined columns
fn parse_column_range(columns: &[String]) -> Result<Vec<usize>, String> {
    if columns.is_empty() {
        return Err("No feature columns specified".to_string());
    }

    let mut indices = vec![];
    for col in columns {
        let col = col.trim();
        
        // Try to parse as range "ch0:ch11"
        if let Some(colon_pos) = col.find(':') {
            let start_str = &col[..colon_pos].trim_end_matches(|c: char| !c.is_alphanumeric());
            let end_str = &col[colon_pos + 1..];

            if let (Ok(start), Ok(end)) = (start_str.parse::<usize>(), end_str.parse::<usize>()) {
                for i in start..=end {
                    indices.push(i);
                }
            } else {
                // Try to extract numbers from strings like "ch0" and "ch11"
                let start_num = extract_trailing_number(start_str)?;
                let end_num = extract_trailing_number(end_str)?;
                for i in start_num..=end_num {
                    indices.push(i);
                }
            }
        } else {
            // Try to parse as direct column index (numeric)
            if let Ok(idx) = col.parse::<usize>() {
                indices.push(idx);
            } else {
                // Try to extract trailing number from column names like "ch0", "ch1", etc.
                if let Ok(idx) = extract_trailing_number(col) {
                    indices.push(idx);
                } else {
                    return Err(format!("Cannot parse column spec: {}", col));
                }
            }
        }
    }

    if indices.is_empty() {
        return Err("No valid feature columns parsed".to_string());
    }

    Ok(indices)
}

fn extract_trailing_number(s: &str) -> Result<usize, String> {
    s.chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
        .parse::<usize>()
        .map_err(|_| format!("Cannot extract number from: {}", s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range() {
        let cols = vec!["ch0:ch11".to_string()];
        let indices = parse_column_range(&cols).unwrap();
        assert_eq!(indices.len(), 12);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[11], 11);
    }

    #[test]
    fn test_parse_direct() {
        let cols = vec!["0".to_string(), "1".to_string(), "5".to_string()];
        let indices = parse_column_range(&cols).unwrap();
        assert_eq!(indices, vec![0, 1, 5]);
    }
}
