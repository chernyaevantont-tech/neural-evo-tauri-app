/// Shape Inference Service
/// 
/// Provides unified shape calculation for datasets across all stream types.
/// This is the single source of truth for determining input/output dimensions.

use std::collections::HashMap;
use std::path::Path;

use crate::csv_loader::CsvDatasetLoader;
use crate::dtos::{CsvDatasetDef, DataLocatorDef, DataStream, DataType};

#[derive(Debug, Clone)]
pub struct ShapeInference;

#[derive(Debug, Clone, Serialize)]
pub struct StreamShapeInfo {
    pub stream_id: String,
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
    pub num_classes: Option<usize>,
    pub inferred_data_type: String, // "Image" | "TemporalSequence" | "Vector" | "Categorical" | "Text"
    pub warnings: Vec<String>,
}

use serde::Serialize;

impl ShapeInference {
    /// Infer input shape for a single stream (Input role)
    ///
    /// # Arguments
    /// * `stream` - The data stream configuration
    /// * `root_path` - Root path for resolving file paths
    ///
    /// # Returns
    /// A tuple of (input_shape, inferred_data_type, warnings)
    pub fn infer_input_shape(
        stream: &DataStream,
        root_path: &Path,
    ) -> Result<(Vec<usize>, String, Vec<String>), String> {
        if stream.role != "Input" {
            return Err("Input shape inference only for Input role streams".to_string());
        }

        let mut warnings = vec![];

        match &stream.locator {
            DataLocatorDef::CsvDataset(csv_def) => {
                ShapeInference::infer_csv_input_shape(csv_def, &stream.data_type, root_path, &mut warnings)
            }

            DataLocatorDef::GlobPattern { pattern: _ } => {
                // For images, return shape from tensor_shape
                if stream.data_type == DataType::Image {
                    if stream.tensor_shape.is_empty() {
                        return Err(
                            "Image stream must specify tensorShape [H, W, C]".to_string(),
                        );
                    }
                    Ok((stream.tensor_shape.clone(), "Image".to_string(), warnings))
                } else {
                    Err(
                        "GlobPattern locator with non-Image data type not yet supported"
                            .to_string(),
                    )
                }
            }

            DataLocatorDef::FolderMapping => {
                // For images
                if stream.data_type == DataType::Image {
                    if stream.tensor_shape.is_empty() {
                        return Err(
                            "Image stream with FolderMapping must specify tensorShape [H, W, C]"
                                .to_string(),
                        );
                    }
                    Ok((stream.tensor_shape.clone(), "Image".to_string(), warnings))
                } else {
                    Err("FolderMapping only supports Image data type".to_string())
                }
            }

            _ => Err(format!(
                "Unsupported locator for input shape inference: {:?}",
                stream.locator
            )),
        }
    }

    /// Infer CSV input shape based on data type (which determines sample mode)
    /// 
    /// IMPORTANT: This function resolves the Phase 3 design issue where users had to 
    /// configure both DataType AND SampleMode separately. Now:
    /// 
    /// **DataType is the source of truth**, not SampleMode:
    /// - If you set DataType = TemporalSequence → we FORCE temporal_window mode
    /// - If you set DataType = Vector → we FORCE row mode
    /// 
    /// The csv_def.sample_mode field is now ignored for shape calculation.
    /// Users only need to set DataType, and the shape inference automatically 
    /// uses the correct sampling strategy.
    /// 
    /// # Example
    /// ```
    /// User sets:  DataType="TemporalSequence", window_size=50, features=12
    /// Result:     input_shape = [50, 12]  (temporal window forced automatically)
    /// 
    /// User sets:  DataType="Vector", features=12
    /// Result:     input_shape = [12]  (row mode forced automatically)
    /// ```
    fn infer_csv_input_shape(
        csv_def: &CsvDatasetDef,
        data_type: &DataType,
        _root_path: &Path,
        warnings: &mut Vec<String>,
    ) -> Result<(Vec<usize>, String, Vec<String>), String> {
        // Determine effective sample mode from data_type
        // This eliminates the confusing dual-configuration (dataType + sampleMode)
        let effective_mode = match data_type {
            DataType::TemporalSequence => "temporal_window",
            DataType::Vector => "row",
            DataType::Categorical => "row", // Target should not reach here
            DataType::Image => "row",         // Target should not reach here
            DataType::Text => "row",          // Default to row
        };

        // For temporal window mode: [window_size, num_features]
        if effective_mode == "temporal_window" {
            let window_size = csv_def
                .window_size
                .ok_or("Temporal window mode requires window_size")?;
            let num_features = csv_def.feature_columns.len();

            if num_features == 0 {
                return Err("CSV temporal stream must have feature_columns".to_string());
            }

            return Ok((
                vec![window_size, num_features],
                "TemporalSequence".to_string(),
                warnings.clone(),
            ));
        }

        // For row mode: [num_features]
        if effective_mode == "row" {
            let num_features = csv_def.feature_columns.len();

            if num_features == 0 {
                // Could be a target-only stream with no features
                return Ok((vec![], "Vector".to_string(), warnings.clone()));
            }

            return Ok((
                vec![num_features],
                "Vector".to_string(),
                warnings.clone(),
            ));
        }

        // Should not reach here if data_type is properly set
        Err(format!(
            "Unexpected effective mode: {} for data_type: {:?}",
            effective_mode, data_type
        ))
    }

    /// Infer output shape for a Target stream
    ///
    /// # Arguments
    /// * `stream` - The target stream configuration
    /// * `num_classes` - Number of distinct classes discovered in target column
    ///
    /// # Returns
    /// output_shape vector
    pub fn infer_output_shape(
        stream: &DataStream,
        num_classes: usize,
    ) -> Result<Vec<usize>, String> {
        if stream.role != "Target" {
            return Err("Output shape only for Target role streams".to_string());
        }

        match stream.data_type {
            DataType::Categorical => {
                // For classification: output is [num_classes]
                if num_classes == 0 {
                    return Err("Target stream has no classes discovered".to_string());
                }
                Ok(vec![num_classes])
            }
            DataType::Vector => {
                // For regression on targets: output is [num_classes] (single or multi-output regression)
                Ok(vec![num_classes])
            }
            _ => Err(format!(
                "Unsupported Target data type for output shape: {:?}",
                stream.data_type
            )),
        }
    }

    /// Validate sample alignment across multiple streams
    ///
    /// When one stream is in temporal_window mode, all other streams must have
    /// their sample count adjusted to match the temporal stream's sample count.
    ///
    /// # Arguments
    /// * `csv_loaders` - Map of CSV path -> loaded CsvDatasetLoader
    /// * `streams` - All streams in the profile
    ///
    /// # Returns
    /// List of valid sample IDs that have all streams represented
    pub fn validate_sample_alignment(
        csv_loaders: &HashMap<String, CsvDatasetLoader>,
        streams: &[DataStream],
    ) -> Result<Vec<String>, String> {
        // Find if any stream is in temporal_window mode
        let mut temporal_window_count: Option<usize> = None;

        for stream in streams {
            if let DataLocatorDef::CsvDataset(csv_def) = &stream.locator {
                if csv_def.sample_mode == "temporal_window" {
                    if let Some(loader) = csv_loaders.get(&csv_def.csv_path) {
                        temporal_window_count = Some(loader.num_samples);
                        break;
                    }
                }
            }
        }

        // If no temporal window, all samples are valid up to the minimum count
        if temporal_window_count.is_none() {
            let mut min_count = usize::MAX;

            for stream in streams {
                if let DataLocatorDef::CsvDataset(csv_def) = &stream.locator {
                    if let Some(loader) = csv_loaders.get(&csv_def.csv_path) {
                        min_count = min_count.min(loader.num_samples);
                    }
                }
            }

            if min_count == usize::MAX {
                return Err("No CSV loaders found to validate alignment".to_string());
            }

            // Return sample IDs as "0", "1", etc.
            return Ok((0..min_count).map(|i| i.to_string()).collect());
        }

        // Temporal window exists: align all other streams to this count
        let temporal_count = temporal_window_count.unwrap();
        Ok((0..temporal_count).map(|i| i.to_string()).collect())
    }

    /// Check if two streams have compatible sample counts
    pub fn check_sample_compatibility(
        count1: usize,
        count2: usize,
        is_temporal: bool,
    ) -> (bool, Vec<String>) {
        let mut warnings = vec![];

        if is_temporal && count1 < count2 {
            warnings.push(format!(
                "Temporal stream ({} samples) has fewer than comparison stream ({} samples). \
                 Alignment will truncate.",
                count1, count2
            ));
            (true, warnings)
        } else if count1 != count2 && !is_temporal {
            warnings.push(format!(
                "Sample count mismatch: {} vs {}. Alignment will drop mismatched samples.",
                count1, count2
            ));
            (true, warnings)
        } else {
            (true, warnings)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_temporal_input_shape() {
        let csv_def = CsvDatasetDef {
            csv_path: "test.csv".to_string(),
            has_headers: true,
            sample_mode: "temporal_window".to_string(),
            feature_columns: vec![
                "ch0".to_string(),
                "ch1".to_string(),
                "ch2".to_string(),
                "ch3".to_string(),
                "ch4".to_string(),
                "ch5".to_string(),
                "ch6".to_string(),
                "ch7".to_string(),
                "ch8".to_string(),
                "ch9".to_string(),
                "ch10".to_string(),
                "ch11".to_string(),
            ],
            target_column: "gesture".to_string(),
            window_size: Some(50),
            window_stride: None,
            preprocessing: crate::dtos::CsvPreprocessingConfig {
                normalization: "none".to_string(),
                handle_missing: "skip".to_string(),
            },
        };

        let mut warnings = vec![];
        let (shape, data_type, _) =
            ShapeInference::infer_csv_input_shape(&csv_def, &DataType::TemporalSequence, Path::new("."), &mut warnings)
                .expect("Should infer shape");

        assert_eq!(shape, vec![50, 12]);
        assert_eq!(data_type, "TemporalSequence");
    }

    #[test]
    fn test_infer_vector_input_shape() {
        let csv_def = CsvDatasetDef {
            csv_path: "iris.csv".to_string(),
            has_headers: true,
            sample_mode: "row".to_string(),
            feature_columns: vec![
                "sepal_length".to_string(),
                "sepal_width".to_string(),
                "petal_length".to_string(),
                "petal_width".to_string(),
            ],
            target_column: "species".to_string(),
            window_size: None,
            window_stride: None,
            preprocessing: crate::dtos::CsvPreprocessingConfig {
                normalization: "none".to_string(),
                handle_missing: "skip".to_string(),
            },
        };

        let mut warnings = vec![];
        let (shape, data_type, _) =
            ShapeInference::infer_csv_input_shape(&csv_def, &DataType::Vector, Path::new("."), &mut warnings)
                .expect("Should infer shape");

        assert_eq!(shape, vec![4]);
        assert_eq!(data_type, "Vector");
    }

    #[test]
    fn test_infer_output_shape_categorical() {
        let stream = DataStream {
            id: "target_stream".to_string(),
            alias: "Target".to_string(),
            role: "Target".to_string(),
            data_type: DataType::Categorical,
            tensor_shape: vec![],
            num_classes: None,
            locator: DataLocatorDef::CsvDataset(CsvDatasetDef {
                csv_path: "test.csv".to_string(),
                has_headers: true,
                sample_mode: "row".to_string(),
                feature_columns: vec![],
                target_column: "gesture".to_string(),
                window_size: None,
                window_stride: None,
                preprocessing: crate::dtos::CsvPreprocessingConfig {
                    normalization: "none".to_string(),
                    handle_missing: "skip".to_string(),
                },
            }),
            preprocessing: None,
        };

        let output_shape =
            ShapeInference::infer_output_shape(&stream, 5).expect("Should infer output shape");
        assert_eq!(output_shape, vec![5]);
    }

    #[test]
    fn test_infer_output_shape_zero_classes_error() {
        let stream = DataStream {
            id: "target_stream".to_string(),
            alias: "Target".to_string(),
            role: "Target".to_string(),
            data_type: DataType::Categorical,
            tensor_shape: vec![],
            num_classes: None,
            locator: DataLocatorDef::CsvDataset(CsvDatasetDef {
                csv_path: "test.csv".to_string(),
                has_headers: true,
                sample_mode: "row".to_string(),
                feature_columns: vec![],
                target_column: "gesture".to_string(),
                window_size: None,
                window_stride: None,
                preprocessing: crate::dtos::CsvPreprocessingConfig {
                    normalization: "none".to_string(),
                    handle_missing: "skip".to_string(),
                },
            }),
            preprocessing: None,
        };

        let result = ShapeInference::infer_output_shape(&stream, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_compatibility_temporal() {
        let (compat, warnings) = ShapeInference::check_sample_compatibility(100, 200, true);
        assert!(compat);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_sample_compatibility_row_mode_match() {
        let (compat, warnings) = ShapeInference::check_sample_compatibility(100, 100, false);
        assert!(compat);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_sample_compatibility_row_mode_mismatch() {
        let (compat, warnings) = ShapeInference::check_sample_compatibility(100, 200, false);
        assert!(compat);
        assert!(!warnings.is_empty());
    }
}
