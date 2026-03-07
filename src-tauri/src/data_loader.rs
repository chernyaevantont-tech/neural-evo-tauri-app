use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::dtos::{DataLocatorDef, DataType, DatasetProfile};
use crate::entities::DynamicTensor;

type Backend = Autodiff<Wgpu>;

// Struct to hold loaded data for a single specific sample
pub struct SampleData {
    pub id: String,
    // Each index corresponds to the stream index in the DatasetProfile
    pub stream_tensors: HashMap<usize, DynamicTensor<Backend>>,
}

pub struct DataLoader {
    profile: DatasetProfile,
    root_path: PathBuf,
    // Stream ID -> Map of SampleID -> FilePath/LocatorData
    stream_files: HashMap<String, HashMap<String, String>>,
    pub valid_sample_ids: Vec<String>,
    pub stream_classes: HashMap<usize, usize>,
}

impl DataLoader {
    pub fn new(profile: DatasetProfile) -> Result<Self, String> {
        let root_path_str = profile
            .source_path
            .clone()
            .ok_or_else(|| "No source path".to_string())?;
        let root_path = PathBuf::from(&root_path_str);
        if !root_path.exists() {
            return Err(format!("Root path does not exist: {}", root_path_str));
        }

        let mut loader = DataLoader {
            profile: profile.clone(),
            root_path,
            stream_files: HashMap::new(),
            valid_sample_ids: Vec::new(),
            stream_classes: HashMap::new(),
        };

        loader.init_locators()?;
        Ok(loader)
    }

    fn init_locators(&mut self) -> Result<(), String> {
        let mut all_sample_ids: Option<HashSet<String>> = None;

        for stream in &self.profile.streams {
            let mut stream_map = HashMap::new();
            match &stream.locator {
                DataLocatorDef::GlobPattern { pattern } => {
                    let found = collect_glob_ids(&self.root_path, pattern);
                    for (id, path) in found {
                        stream_map.insert(id, path.to_string_lossy().to_string());
                    }
                }
                DataLocatorDef::FolderMapping => {
                    // Collect class names from parent folder of existing valid IDs
                    // This relies on having an anchor stream first.
                    // For simplicity, we assume FolderMapping is not the FIRST stream unless we auto-discover.
                    // If we need auto-discover, we do a generic image search.
                    let found = collect_glob_ids(&self.root_path, "**/*.{jpg,png,jpeg,webp}");
                    for (id, path) in found {
                        if let Some(parent) = path.parent() {
                            if let Some(folder_name) = parent.file_name() {
                                stream_map.insert(id, folder_name.to_string_lossy().to_string());
                            }
                        }
                    }
                }
                DataLocatorDef::MasterIndex {
                    index_path,
                    key_field,
                    value_field,
                    has_headers,
                } => {
                    // Parse CSV and map key_field to value_field
                    let full_path = self.root_path.join(index_path);
                    if let Ok(mut rdr) = csv::ReaderBuilder::new()
                        .has_headers(*has_headers)
                        .from_path(&full_path)
                    {
                        let headers = rdr.headers().unwrap_or(&csv::StringRecord::new()).clone();

                        // If has_headers is false, we don't have headers to search strings against
                        let key_idx = if key_field.is_empty() {
                            None
                        } else if *has_headers {
                            headers.iter().position(|h| h == key_field)
                        } else {
                            key_field.trim().parse::<usize>().ok()
                        };

                        // If value_field is empty or "*", we take all other columns
                        let val_indices: Vec<usize> =
                            if value_field.is_empty() || value_field == "*" {
                                if *has_headers {
                                    headers
                                        .iter()
                                        .enumerate()
                                        .filter_map(
                                            |(i, h)| if h != key_field { Some(i) } else { None },
                                        )
                                        .collect()
                                } else {
                                    // Assume we have at least one record to determine length
                                    if let Some(Ok(_first_rec)) = rdr.records().next() {
                                        // We defer to inside the loop — gather all except key_idx
                                        vec![]
                                    } else {
                                        vec![]
                                    }
                                }
                            } else {
                                // Specify exact indices
                                if *has_headers {
                                    let fields: Vec<&str> =
                                        value_field.split(',').map(|s| s.trim()).collect();
                                    headers
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(i, h)| {
                                            if fields.contains(&h) { Some(i) } else { None }
                                        })
                                        .collect()
                                } else {
                                    value_field
                                        .split(',')
                                        .filter_map(|s| s.trim().parse::<usize>().ok())
                                        .collect()
                                }
                            };

                        // Recreate reader to parse from start (since headers/.records().next() consumed state)
                        let mut rdr = csv::ReaderBuilder::new()
                            .has_headers(*has_headers)
                            .from_path(&full_path)
                            .unwrap();

                        for (row_idx, result) in rdr.records().enumerate() {
                            if let Ok(record) = result {
                                let key = if let Some(k) = key_idx {
                                    record.get(k).map(|s| s.to_string())
                                } else {
                                    Some(row_idx.to_string())
                                };

                                if let Some(k_str) = key {
                                    let mut vals = Vec::new();

                                    // If val_indices is empty but value_field is "*", gather all except key_idx
                                    let actual_val_indices = if val_indices.is_empty()
                                        && (value_field.is_empty() || value_field == "*")
                                    {
                                        (0..record.len())
                                            .filter(|&i| Some(i) != key_idx)
                                            .collect::<Vec<usize>>()
                                    } else {
                                        val_indices.clone()
                                    };

                                    for v_idx in actual_val_indices {
                                        if let Some(val) = record.get(v_idx) {
                                            vals.push(val.to_string());
                                        }
                                    }
                                    stream_map.insert(k_str, vals.join(","));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            let current_ids: HashSet<String> = stream_map.keys().cloned().collect();
            if let Some(ref existing) = all_sample_ids {
                if !current_ids.is_empty() {
                    all_sample_ids = Some(existing.intersection(&current_ids).cloned().collect());
                }
            } else if !current_ids.is_empty() {
                all_sample_ids = Some(current_ids);
            }

            self.stream_files.insert(stream.id.clone(), stream_map);
        }

        self.valid_sample_ids = all_sample_ids.unwrap_or_default().into_iter().collect();
        self.valid_sample_ids.sort(); // For determinism

        // Generate categorical mappings for FolderMapping or Categorical types
        for (idx, stream) in self.profile.streams.iter().enumerate() {
            if let DataType::Categorical = stream.data_type {
                // Build vocabulary
                let mut classes = HashSet::new();
                if let Some(map) = self.stream_files.get(&stream.id) {
                    for class_val in map.values() {
                        classes.insert(class_val.clone());
                    }
                }
                let mut sorted_classes: Vec<String> = classes.into_iter().collect();
                sorted_classes.sort();

                self.stream_classes.insert(idx, sorted_classes.len());

                let mut class_to_id = HashMap::new();
                for (idx_class, c) in sorted_classes.into_iter().enumerate() {
                    class_to_id.insert(c, idx_class as f32);
                }

                // Store this mapping somewhere, for now we will just re-compute per sample using a string cache
                // Actually, let's just pre-compute the f32 values in the stream_map
                if let Some(map) = self.stream_files.get_mut(&stream.id) {
                    for val in map.values_mut() {
                        if let Some(&float_val) = class_to_id.get(val) {
                            *val = float_val.to_string();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn load_sample(&self, sample_id: &str, device: &WgpuDevice) -> Result<SampleData, String> {
        let mut tensors = HashMap::new();

        for (idx, stream) in self.profile.streams.iter().enumerate() {
            let locator_val = self
                .stream_files
                .get(&stream.id)
                .and_then(|m| m.get(sample_id))
                .cloned()
                .unwrap_or_default();

            match stream.data_type {
                DataType::Image => {
                    // locator_val is file path
                    let img_path = PathBuf::from(&locator_val);
                    if let Ok(img) = image::open(&img_path) {
                        let mut target_w = 64;
                        let mut target_h = 64;
                        let mut grayscale = false;

                        if let Some(prep) = &stream.preprocessing {
                            if let Some(vision) = &prep.vision {
                                if vision.resize.len() == 2 {
                                    target_w = vision.resize[0];
                                    target_h = vision.resize[1];
                                }
                                grayscale = vision.grayscale;
                            }
                        }

                        let img = img.resize_exact(
                            target_w,
                            target_h,
                            image::imageops::FilterType::Triangle,
                        );
                        let channels = if grayscale { 1 } else { 3 };
                        let mut pixels =
                            Vec::with_capacity((target_w * target_h) as usize * channels);

                        if grayscale {
                            let img_gray = img.to_luma8();
                            for y in 0..target_h {
                                for x in 0..target_w {
                                    pixels.push(img_gray.get_pixel(x, y)[0] as f32 / 255.0);
                                }
                            }
                        } else {
                            let img_rgb = img.to_rgb8();
                            for y in 0..target_h {
                                for x in 0..target_w {
                                    for c in 0..3 {
                                        pixels.push(img_rgb.get_pixel(x, y)[c] as f32 / 255.0);
                                    }
                                }
                            }
                        }

                        let tensor_1d =
                            Tensor::<Backend, 1>::from_floats(pixels.as_slice(), device);
                        let tensor_4d = tensor_1d.reshape([
                            1,
                            channels as usize,
                            target_h as usize,
                            target_w as usize,
                        ]);
                        tensors.insert(idx, DynamicTensor::Dim4(tensor_4d));
                    }
                }
                DataType::Vector => {
                    // Usually from CSV or MasterIndex
                    // For now, assume locator_val contains a comma-separated list of floats, or a single float
                    let vals: Vec<f32> = locator_val
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();

                    if !vals.is_empty() {
                        let tensor = Tensor::<Backend, 1>::from_floats(vals.as_slice(), device);
                        let tensor_2d = tensor.reshape([1, vals.len()]);
                        tensors.insert(idx, DynamicTensor::Dim2(tensor_2d));
                    }
                }
                DataType::Categorical => {
                    // locator_val is the string representation of the class ID (f32) set during init
                    if let Ok(val) = locator_val.parse::<f32>() {
                        let tensor = Tensor::<Backend, 2>::from_data([[val]], device);
                        tensors.insert(idx, DynamicTensor::Dim2(tensor));
                    }
                }
                DataType::Text => {
                    // Not fully implemented yet
                }
            }
        }

        Ok(SampleData {
            id: sample_id.to_string(),
            stream_tensors: tensors,
        })
    }
}

// Helpers reused from scan_dataset
fn collect_glob_ids(root: &Path, pattern: &str) -> HashMap<String, std::path::PathBuf> {
    let full_pattern = root.join(pattern).to_string_lossy().to_string();
    let full_pattern = full_pattern.replace('\\', "/");
    let mut map = HashMap::new();
    if let Ok(paths) = glob::glob(&full_pattern) {
        for entry in paths.flatten() {
            if entry.is_file() {
                if let Ok(relative) = entry.strip_prefix(root) {
                    let id = relative
                        .with_extension("")
                        .to_string_lossy()
                        .replace('\\', "/");
                    map.insert(id, entry);
                }
            }
        }
    }
    map
}
