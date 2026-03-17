use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::dtos::{DataLocatorDef, DataType, DatasetProfile};
use crate::entities::DynamicTensor;
use crate::csv_loader::CsvDatasetLoader;

type Backend = Autodiff<Wgpu>;

#[derive(serde::Serialize)]
pub struct CacheResult {
    pub total_cached: usize,
    pub total_dropped: usize,
    pub dropped_sample_ids: Vec<String>,
    pub class_counts: HashMap<String, usize>,
}

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
    pub stream_files: HashMap<String, HashMap<String, String>>,
    pub valid_sample_ids: Vec<String>,
    pub stream_classes: HashMap<usize, usize>,
    pub app_data_dir: Option<PathBuf>,
    // Cache for CsvDatasetLoader instances (Stream ID -> Loader)
    #[allow(dead_code)]
    csv_loaders: HashMap<String, CsvDatasetLoader>,
}

impl DataLoader {
    pub fn new(profile: DatasetProfile, app_data_dir: Option<PathBuf>) -> Result<Self, String> {
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
            app_data_dir,
            csv_loaders: HashMap::new(),
        };

        loader.init_locators()?;

        // If a cache was built, filter valid_sample_ids to only include cached samples.
        // This removes corrupt/unreadable images discovered during cache building.
        if let Some(ref app_data) = loader.app_data_dir {
            let cache_base = app_data.join("datasets_cache").join(&loader.profile.id);
            // Find any image stream whose cache dir exists
            for stream in &loader.profile.streams {
                if let DataType::Image = stream.data_type {
                    let stream_cache_dir = cache_base.join(&stream.id);
                    if stream_cache_dir.exists() {
                        let before = loader.valid_sample_ids.len();
                        loader.valid_sample_ids.retain(|sample_id| {
                            let cache_path = stream_cache_dir
                                .join(format!("{}.bin", sample_id.replace("/", "_")));
                            cache_path.exists()
                        });
                        let dropped = before - loader.valid_sample_ids.len();
                        if dropped > 0 {
                            println!(
                                "  Cache filter: removed {} uncached samples ({} -> {} valid)",
                                dropped,
                                before,
                                loader.valid_sample_ids.len()
                            );
                        }
                    }
                }
            }
        }

        Ok(loader)
    }

    fn init_locators(&mut self) -> Result<(), String> {
        let mut all_sample_ids: Option<HashSet<String>> = None;
        println!(
            "DataLoader init_locators start... Root: {:?}",
            self.root_path
        );

        for stream in &self.profile.streams {
            let mut stream_map = HashMap::new();
            println!(
                "  Processing stream: ID={}, DataType={:?}",
                stream.id, stream.data_type
            );

            match &stream.locator {
                DataLocatorDef::GlobPattern { pattern } => {
                    println!("    Locator: GlobPattern, pattern: {}", pattern);
                    let found = collect_glob_ids(&self.root_path, pattern);
                    for (id, path) in found {
                        stream_map.insert(id, path.to_string_lossy().to_string());
                    }
                    println!("    GlobPattern found {} samples", stream_map.len());
                }
                DataLocatorDef::FolderMapping => {
                    println!("    Locator: FolderMapping");
                    // We need to iterate extensions like lib.rs to fix the {} issue that glob crate does not support
                    let extensions = ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "csv", "txt"];
                    for ext in &extensions {
                        let pattern = format!("**/*.{}", ext);
                        let found = collect_glob_ids(&self.root_path, &pattern);
                        for (id, path) in found {
                            match stream.data_type {
                                DataType::Categorical => {
                                    if let Some(parent) = path.parent() {
                                        if let Some(folder_name) = parent.file_name() {
                                            stream_map.insert(
                                                id,
                                                folder_name.to_string_lossy().to_string(),
                                            );
                                        }
                                    }
                                }
                                _ => {
                                    // For Image, Vector, Text we need the absolute path
                                    stream_map.insert(id, path.to_string_lossy().to_string());
                                }
                            }
                        }
                    }
                    println!("    FolderMapping found {} samples", stream_map.len());
                }
                DataLocatorDef::MasterIndex {
                    index_path,
                    key_field,
                    value_field,
                    has_headers,
                } => {
                    println!(
                        "    Locator: MasterIndex, index: {}, key: {}, val: {}",
                        index_path, key_field, value_field
                    );
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
                    println!("    MasterIndex found {} samples", stream_map.len());
                }
                DataLocatorDef::CsvDataset(csv_def) => {
                    println!("    Locator: CsvDataset");
                    
                    match CsvDatasetLoader::init(&self.root_path, csv_def.clone()) {
                        Ok(csv_loader) => {
                            // For each discoverable sample in CSV:
                            // - Row mode: sample IDs are "0", "1", "2", ... row_count
                            // - Temporal mode: sample IDs are "0", "1", ... num_windows
                            
                            for sample_idx in 0..csv_loader.num_samples {
                                let sample_id = sample_idx.to_string();
                                stream_map.insert(sample_id, format!("csv:{}", sample_idx));
                            }
                            
                            // Record discovered classes
                            if stream.role == "Target" {
                                self.stream_classes.insert(
                                    self.profile.streams.iter().position(|s| s.id == stream.id).unwrap_or(0),
                                    csv_loader.discovered_classes.len()
                                );
                            }
                            
                            println!("    CsvDataset found {} samples, {} classes",
                                csv_loader.num_samples,
                                csv_loader.discovered_classes.len()
                            );
                            
                            // Cache the loader for later use
                            self.csv_loaders.insert(stream.id.clone(), csv_loader);
                        }
                        Err(e) => {
                            return Err(format!("Failed to init CSV dataset: {}", e));
                        }
                    }
                }
                _ => {
                    println!("    Locator: Other/Unknown");
                }
            }

            let current_ids: HashSet<String> = stream_map.keys().cloned().collect();
            println!("  Stream mapped {} distinct SampleIDs", current_ids.len());

            if let Some(ref existing) = all_sample_ids {
                // BUG FIX: Intentionally perform intersection even if current_ids is empty!
                // If current_ids is empty, the dataset intersection is fundamentally 0.
                let intersected: HashSet<String> =
                    existing.intersection(&current_ids).cloned().collect();
                println!(
                    "  Intersection size reduced from {} to {}",
                    existing.len(),
                    intersected.len()
                );
                all_sample_ids = Some(intersected);
            } else {
                println!(
                    "  First stream anchor initialized with {} samples",
                    current_ids.len()
                );
                all_sample_ids = Some(current_ids);
            }

            self.stream_files.insert(stream.id.clone(), stream_map);
        }

        self.valid_sample_ids = all_sample_ids.unwrap_or_default().into_iter().collect();
        println!(
            "  Final valid_sample_ids count: {}",
            self.valid_sample_ids.len()
        );
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

                if let Some(map) = self.stream_files.get_mut(&stream.id) {
                    for val in map.values_mut() {
                        if let Some(&float_val) = class_to_id.get(val) {
                            *val = float_val.to_string();
                        }
                    }
                }
            }
        }

        println!("DataLoader init_locators done.");
        Ok(())
    }

    pub fn build_image_cache(&self) -> Result<CacheResult, String> {
        let app_data = self
            .app_data_dir
            .as_ref()
            .ok_or("No AppData directory available to store cache")?;
        let cache_dir = app_data.join("datasets_cache").join(&self.profile.id);

        let mut cached_ids: Vec<String> = Vec::new();
        let mut dropped_ids: Vec<String> = Vec::new();

        for stream in &self.profile.streams {
            if let DataType::Image = stream.data_type {
                let stream_cache_dir = cache_dir.join(&stream.id);
                std::fs::create_dir_all(&stream_cache_dir).map_err(|e| e.to_string())?;

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
                let channels = if grayscale { 1 } else { 3 };

                for sample_id in &self.valid_sample_ids {
                    let locator_val = self
                        .stream_files
                        .get(&stream.id)
                        .and_then(|m| m.get(sample_id))
                        .cloned()
                        .unwrap_or_default();

                    let sanitized_path = locator_val.trim_matches(|c| c == '"' || c == '\'');
                    let img_path = PathBuf::from(sanitized_path);

                    let cache_path =
                        stream_cache_dir.join(format!("{}.bin", sample_id.replace("/", "_")));
                    if cache_path.exists() {
                        cached_ids.push(sample_id.clone());
                        continue; // Already processed
                    }

                    if let Ok(img) = image::open(&img_path) {
                        let img = img.resize_exact(
                            target_w,
                            target_h,
                            image::imageops::FilterType::Triangle,
                        );
                        let mut pixels: Vec<f32> =
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
                            for c in 0..3 {
                                for y in 0..target_h {
                                    for x in 0..target_w {
                                        pixels.push(img_rgb.get_pixel(x, y)[c] as f32 / 255.0);
                                    }
                                }
                            }
                        }

                        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
                        if let Err(e) = std::fs::write(&cache_path, bytes) {
                            eprintln!("Failed to write cache for {}: {}", sample_id, e);
                            dropped_ids.push(sample_id.clone());
                        } else {
                            cached_ids.push(sample_id.clone());
                        }
                    } else {
                        eprintln!(
                            "AoT Cache Builder failed to decode image for {}: {:?}",
                            sample_id, img_path
                        );
                        dropped_ids.push(sample_id.clone());
                    }
                }
            }
        }

        // Build per-class counts from sample IDs (e.g. "Cat/1234" -> class "Cat")
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for id in &cached_ids {
            let class_name = id.split('/').next().unwrap_or("Unknown").to_string();
            *class_counts.entry(class_name).or_insert(0) += 1;
        }

        Ok(CacheResult {
            total_cached: cached_ids.len(),
            total_dropped: dropped_ids.len(),
            dropped_sample_ids: dropped_ids,
            class_counts,
        })
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
                    let channels = if grayscale { 1 } else { 3 };
                    let expected_len = (target_w * target_h) as usize * channels;

                    // CHECK CACHE FIRST
                    let mut cached_pixels = None;
                    let mut cache_dir_exists = false;
                    if let Some(app_data) = &self.app_data_dir {
                        let stream_cache_dir = app_data
                            .join("datasets_cache")
                            .join(&self.profile.id)
                            .join(&stream.id);
                        cache_dir_exists = stream_cache_dir.exists();

                        let cache_path =
                            stream_cache_dir.join(format!("{}.bin", sample_id.replace("/", "_")));

                        if let Ok(bytes) = std::fs::read(&cache_path) {
                            let floats: &[f32] = bytemuck::cast_slice(&bytes);
                            if floats.len() == expected_len {
                                cached_pixels = Some(floats.to_vec());
                            }
                        }
                    }

                    let pixels = if let Some(p) = cached_pixels {
                        p
                    } else if cache_dir_exists {
                        // Cache was built but this sample has no .bin -> it was corrupt
                        return Err(format!(
                            "Sample '{}' was not cached (corrupt/unreadable during cache build)",
                            sample_id
                        ));
                    } else {
                        let sanitized_path = locator_val.trim_matches(|c| c == '"' || c == '\'');
                        let img_path = PathBuf::from(sanitized_path);

                        match image::open(&img_path) {
                            Ok(img) => {
                                let img = img.resize_exact(
                                    target_w,
                                    target_h,
                                    image::imageops::FilterType::Triangle,
                                );
                                let mut raw_pixels = Vec::with_capacity(expected_len);

                                if grayscale {
                                    let img_gray = img.to_luma8();
                                    for y in 0..target_h {
                                        for x in 0..target_w {
                                            raw_pixels
                                                .push(img_gray.get_pixel(x, y)[0] as f32 / 255.0);
                                        }
                                    }
                                } else {
                                    let img_rgb = img.to_rgb8();
                                    for c in 0..3 {
                                        for y in 0..target_h {
                                            for x in 0..target_w {
                                                raw_pixels.push(
                                                    img_rgb.get_pixel(x, y)[c] as f32 / 255.0,
                                                );
                                            }
                                        }
                                    }
                                }
                                raw_pixels
                            }
                            Err(e) => {
                                return Err(format!(
                                    "Failed to load image for stream {} at path '{}': {:?}",
                                    stream.id,
                                    img_path.display(),
                                    e
                                ));
                            }
                        }
                    };

                    let tensor_1d = Tensor::<Backend, 1>::from_floats(pixels.as_slice(), device);
                    let tensor_4d = tensor_1d.reshape([
                        1,
                        channels as usize,
                        target_h as usize,
                        target_w as usize,
                    ]);
                    tensors.insert(idx, DynamicTensor::Dim4(tensor_4d));
                }
                DataType::Vector => {
                    let vals: Vec<f32> = locator_val
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();

                    if !vals.is_empty() {
                        let tensor = Tensor::<Backend, 1>::from_floats(vals.as_slice(), device);
                        let tensor_2d = tensor.reshape([1, vals.len()]);
                        tensors.insert(idx, DynamicTensor::Dim2(tensor_2d));
                    } else {
                        return Err(format!(
                            "Failed to parse vector (CSV) values for stream {} from string: '{}'",
                            stream.id, locator_val
                        ));
                    }
                }
                DataType::Categorical => match locator_val.parse::<f32>() {
                    Ok(val) => {
                        let tensor = Tensor::<Backend, 2>::from_data([[val]], device);
                        tensors.insert(idx, DynamicTensor::Dim2(tensor));
                    }
                    Err(e) => {
                        return Err(format!(
                            "Failed to parse categorical float for stream {} from string: '{}': {:?}",
                            stream.id, locator_val, e
                        ));
                    }
                },
                DataType::Text => {
                    // Not fully implemented yet
                }
                DataType::TemporalSequence => {
                    // Load temporal window from CSV dataset
                    if locator_val.starts_with("csv:") {
                        let sample_idx_str = &locator_val[4..];
                        let sample_idx = sample_idx_str.parse::<usize>()
                            .map_err(|_| format!("Invalid CSV sample index: {}", sample_idx_str))?;
                        
                        // Get the cached CSV loader for this stream
                        if let Some(csv_loader) = self.csv_loaders.get(&stream.id) {
                            match csv_loader.load_sample(sample_idx, device) {
                                Ok((tensor, _label)) => {
                                    tensors.insert(idx, tensor);
                                }
                                Err(e) => {
                                    return Err(format!("Failed to load CSV sample {}: {}", sample_idx, e));
                                }
                            }
                        } else {
                            return Err(format!("No CSV loader found for stream {}", stream.id));
                        }
                    } else {
                        return Err(format!("Invalid temporal sequence locator format: {}", locator_val));
                    }
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
