// CSV Dataset Integration Point Reference
// 
// This file documents the minimal changes needed for Phase 1 completion
// and shows how CSV support should be integrated into the main DataLoader flow.

/*
PHASE 1 INTEGRATION CHECKLIST:

1. In data_loader.rs::init_locators(), add CSV case to the match statement:

    DataLocatorDef::CsvDataset(csv_def) => {
        println!("    Locator: CsvDataset");
        
        match crate::csv_loader::CsvDatasetLoader::init(&self.root_path, csv_def.clone()) {
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
                        stream_index,
                        csv_loader.discovered_classes.len()
                    );
                }
                
                println!("    CsvDataset found {} samples, {} classes",
                    csv_loader.num_samples,
                    csv_loader.discovered_classes.len()
                );
            }
            Err(e) => {
                return Err(format!("Failed to init CSV dataset: {}", e));
            }
        }
    }

2. In data_loader.rs::load_sample(), add case for TemporalSequence:

    DataType::TemporalSequence => {
        // Parse locator_val to identify which CSV to load from
        // Format: "csv:sample_idx" or similar
        
        if locator_val.starts_with("csv:") {
            let sample_idx_str = &locator_val[4..];
            let sample_idx = sample_idx_str.parse::<usize>()?;
            
            // Call CsvDatasetLoader to load this sample
            let (tensor, _label) = csv_loader.load_sample(sample_idx, device)?;
            tensors.insert(idx, tensor);
        }
    }

3. Cache CsvDatasetLoader instances:
   - Keep reference to csv_loader for each CSV stream
   - Reuse across multiple load_sample calls for efficiency

CURRENT STATE:
✅ CsvDatasetDef struct fully defined in dtos.rs
✅ CsvDatasetLoader fully implemented in csv_loader.rs
✅ TemporalSequence added to DataType enum
✅ CsvDataset variant added to DataLocatorDef enum
⏳ Integration points identified above (straighforward copy-paste)

READY TO TRANSITION TO PHASE 2:
Frontend UI components can now be built independently.
Backend integration can proceed in parallel.
*/
