use crate::device_profiles;
use crate::dtos::{
    CreateDeviceTemplateInput, DeviceLibraryImportMode, DeviceTemplateDto, UpdateDeviceTemplatePatch,
};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const STORAGE_ENV_VAR: &str = "NEURAL_EVO_DEVICE_LIBRARY_DIR";
const STORAGE_FILE_NAME: &str = "device_templates.json";

fn current_unix_ms() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

fn get_device_library_dir() -> PathBuf {
    if let Ok(custom_dir) = std::env::var(STORAGE_ENV_VAR) {
        return PathBuf::from(custom_dir);
    }

    let exe_dir = std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from("."))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    exe_dir.join("device_library")
}

fn get_device_library_path() -> PathBuf {
    get_device_library_dir().join(STORAGE_FILE_NAME)
}

fn normalize_name(name: &str) -> String {
    name.trim().to_string()
}

fn normalize_notes(notes: Option<String>) -> Option<String> {
    notes.and_then(|n| {
        let trimmed = n.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_tags(tags: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();

    for tag in tags {
        let cleaned = tag.trim();
        if cleaned.is_empty() {
            continue;
        }

        let key = cleaned.to_lowercase();
        if seen.insert(key) {
            normalized.push(cleaned.to_string());
        }
    }

    normalized
}

fn validate_template_name(name: &str) -> Result<(), String> {
    let len = name.chars().count();
    if !(2..=64).contains(&len) {
        return Err("Template name must have length in range 2..64".to_string());
    }

    Ok(())
}

fn ensure_unique_name(
    templates: &[DeviceTemplateDto],
    name: &str,
    excluded_id: Option<&str>,
) -> Result<(), String> {
    let lowered_name = name.to_lowercase();

    let exists = templates.iter().any(|template| {
        if excluded_id.is_some_and(|excluded| template.id == excluded) {
            return false;
        }
        template.name.to_lowercase() == lowered_name
    });

    if exists {
        return Err(format!("Template name '{}' already exists", name));
    }

    Ok(())
}

fn read_device_templates() -> Result<Vec<DeviceTemplateDto>, String> {
    let path = get_device_library_path();
    if !path.exists() {
        return Ok(Vec::new());
    }

    let raw = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str::<Vec<DeviceTemplateDto>>(&raw).map_err(|e| e.to_string())
}

fn write_device_templates(entries: &[DeviceTemplateDto]) -> Result<(), String> {
    let dir = get_device_library_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let json = serde_json::to_string_pretty(entries).map_err(|e| e.to_string())?;
    fs::write(get_device_library_path(), json).map_err(|e| e.to_string())
}

fn normalize_import_template(mut template: DeviceTemplateDto) -> Result<DeviceTemplateDto, String> {
    template.name = normalize_name(&template.name);
    validate_template_name(&template.name)?;
    device_profiles::validate_constraints(&template.constraints)?;
    template.notes = normalize_notes(template.notes);
    template.tags = normalize_tags(template.tags);

    if template.id.trim().is_empty() {
        template.id = uuid::Uuid::new_v4().to_string();
    }

    let now = current_unix_ms();
    if template.created_at_unix_ms == 0 {
        template.created_at_unix_ms = now;
    }
    if template.updated_at_unix_ms == 0 {
        template.updated_at_unix_ms = now;
    }

    Ok(template)
}

fn ensure_import_uniqueness(templates: &[DeviceTemplateDto]) -> Result<(), String> {
    let mut names = HashSet::new();
    let mut ids = HashSet::new();

    for template in templates {
        let name_key = template.name.to_lowercase();
        if !names.insert(name_key) {
            return Err("Imported library has duplicate names".to_string());
        }

        if !ids.insert(template.id.clone()) {
            return Err("Imported library has duplicate ids".to_string());
        }
    }

    Ok(())
}

pub fn list_device_templates() -> Result<Vec<DeviceTemplateDto>, String> {
    read_device_templates()
}

pub fn create_device_template(input: CreateDeviceTemplateInput) -> Result<DeviceTemplateDto, String> {
    let mut entries = read_device_templates()?;

    let name = normalize_name(&input.name);
    validate_template_name(&name)?;
    ensure_unique_name(&entries, &name, None)?;
    device_profiles::validate_constraints(&input.constraints)?;

    let now = current_unix_ms();
    let created = DeviceTemplateDto {
        id: uuid::Uuid::new_v4().to_string(),
        name,
        constraints: input.constraints,
        notes: normalize_notes(input.notes),
        tags: normalize_tags(input.tags),
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
    };

    entries.push(created.clone());
    write_device_templates(&entries)?;

    Ok(created)
}

pub fn update_device_template(id: String, patch: UpdateDeviceTemplatePatch) -> Result<DeviceTemplateDto, String> {
    let mut entries = read_device_templates()?;
    let idx = entries
        .iter()
        .position(|entry| entry.id == id)
        .ok_or_else(|| format!("Template '{}' not found", id))?;

    let mut updated = entries[idx].clone();

    if let Some(name) = patch.name {
        let name = normalize_name(&name);
        validate_template_name(&name)?;
        ensure_unique_name(&entries, &name, Some(updated.id.as_str()))?;
        updated.name = name;
    }

    if let Some(constraints) = patch.constraints {
        device_profiles::validate_constraints(&constraints)?;
        updated.constraints = constraints;
    }

    if let Some(notes) = patch.notes {
        updated.notes = normalize_notes(Some(notes));
    }

    if let Some(tags) = patch.tags {
        updated.tags = normalize_tags(tags);
    }

    updated.updated_at_unix_ms = current_unix_ms();
    entries[idx] = updated.clone();
    write_device_templates(&entries)?;

    Ok(updated)
}

pub fn delete_device_template(id: String) -> Result<(), String> {
    let mut entries = read_device_templates()?;
    let before = entries.len();
    entries.retain(|entry| entry.id != id);

    if before == entries.len() {
        return Err(format!("Template '{}' not found", id));
    }

    write_device_templates(&entries)
}

pub fn duplicate_device_template(id: String, new_name: String) -> Result<DeviceTemplateDto, String> {
    let mut entries = read_device_templates()?;
    let source = entries
        .iter()
        .find(|entry| entry.id == id)
        .cloned()
        .ok_or_else(|| format!("Template '{}' not found", id))?;

    let normalized_name = normalize_name(&new_name);
    validate_template_name(&normalized_name)?;
    ensure_unique_name(&entries, &normalized_name, None)?;

    let now = current_unix_ms();
    let cloned = DeviceTemplateDto {
        id: uuid::Uuid::new_v4().to_string(),
        name: normalized_name,
        constraints: source.constraints,
        notes: source.notes,
        tags: source.tags,
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
    };

    entries.push(cloned.clone());
    write_device_templates(&entries)?;

    Ok(cloned)
}

pub fn export_device_library(path: String) -> Result<usize, String> {
    let entries = read_device_templates()?;

    let export_path = PathBuf::from(path);
    if let Some(parent) = export_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let json = serde_json::to_string_pretty(&entries).map_err(|e| e.to_string())?;
    fs::write(export_path, json).map_err(|e| e.to_string())?;

    Ok(entries.len())
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum ImportPayload {
    Direct(Vec<DeviceTemplateDto>),
    Wrapped { templates: Vec<DeviceTemplateDto> },
}

pub fn import_device_library(path: String, mode: DeviceLibraryImportMode) -> Result<Vec<DeviceTemplateDto>, String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let parsed = serde_json::from_str::<ImportPayload>(&raw).map_err(|e| e.to_string())?;

    let imported_raw = match parsed {
        ImportPayload::Direct(items) => items,
        ImportPayload::Wrapped { templates } => templates,
    };

    let mut imported = Vec::with_capacity(imported_raw.len());
    for template in imported_raw {
        imported.push(normalize_import_template(template)?);
    }
    ensure_import_uniqueness(&imported)?;

    let result = match mode {
        DeviceLibraryImportMode::Replace => imported,
        DeviceLibraryImportMode::Merge => {
            let mut merged = read_device_templates()?;

            for template in imported {
                let template_name_key = template.name.to_lowercase();
                merged.retain(|existing| {
                    existing.id != template.id && existing.name.to_lowercase() != template_name_key
                });
                merged.push(template);
            }

            merged
        }
    };

    write_device_templates(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn constraints() -> crate::device_profiles::DeviceResourceConstraints {
        crate::device_profiles::DeviceResourceConstraints {
            mops_budget: 1000.0,
            ram_budget_mb: 64.0,
            flash_budget_mb: 128.0,
            max_latency_ms: 30.0,
        }
    }

    fn with_test_storage(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{}-{}", prefix, uuid::Uuid::new_v4()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        unsafe {
            std::env::set_var(STORAGE_ENV_VAR, dir.to_string_lossy().to_string());
        }
        dir
    }

    #[test]
    fn device_library_crud_flow() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        let temp = with_test_storage("device-library-crud");

        let created = create_device_template(CreateDeviceTemplateInput {
            name: "Edge Tiny".to_string(),
            constraints: constraints(),
            notes: Some("Main target".to_string()),
            tags: vec!["edge".to_string(), "tiny".to_string()],
        })
        .expect("create template");

        let updated = update_device_template(
            created.id.clone(),
            UpdateDeviceTemplatePatch {
                name: Some("Edge Tiny v2".to_string()),
                constraints: None,
                notes: Some("Updated".to_string()),
                tags: Some(vec!["edge".to_string(), "v2".to_string()]),
            },
        )
        .expect("update template");

        assert_eq!(updated.name, "Edge Tiny v2");
        assert_eq!(updated.tags, vec!["edge".to_string(), "v2".to_string()]);

        let duplicate = duplicate_device_template(updated.id.clone(), "Edge Tiny v2 Copy".to_string())
            .expect("duplicate template");

        let listed = list_device_templates().expect("list templates");
        assert_eq!(listed.len(), 2);

        delete_device_template(updated.id).expect("delete original");
        let listed_after_delete = list_device_templates().expect("list after delete");
        assert_eq!(listed_after_delete.len(), 1);
        assert_eq!(listed_after_delete[0].id, duplicate.id);

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn device_library_enforces_case_insensitive_unique_names() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        let temp = with_test_storage("device-library-unique");

        create_device_template(CreateDeviceTemplateInput {
            name: "Mobile Low-End".to_string(),
            constraints: constraints(),
            notes: None,
            tags: vec![],
        })
        .expect("create first template");

        let err = create_device_template(CreateDeviceTemplateInput {
            name: "mobile low-end".to_string(),
            constraints: constraints(),
            notes: None,
            tags: vec![],
        })
        .expect_err("must reject duplicate name");

        assert!(err.contains("already exists"));

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn device_library_rejects_non_positive_constraints() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        let temp = with_test_storage("device-library-validation");

        let mut invalid = constraints();
        invalid.max_latency_ms = 0.0;

        let err = create_device_template(CreateDeviceTemplateInput {
            name: "Invalid Device".to_string(),
            constraints: invalid,
            notes: None,
            tags: vec![],
        })
        .expect_err("must reject invalid constraints");

        assert!(err.contains("max_latency_ms"));

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn device_library_import_export_roundtrip_merge_replace() {
        let _guard = TEST_LOCK.lock().expect("test lock");

        let source_storage = with_test_storage("device-library-source");
        create_device_template(CreateDeviceTemplateInput {
            name: "Exported Device".to_string(),
            constraints: constraints(),
            notes: Some("Export me".to_string()),
            tags: vec!["export".to_string()],
        })
        .expect("create source template");

        let export_path = source_storage.join("exports").join("library.json");
        let exported_count = export_device_library(export_path.to_string_lossy().to_string())
            .expect("export library");
        assert_eq!(exported_count, 1);

        let target_storage = with_test_storage("device-library-target");
        let replaced = import_device_library(
            export_path.to_string_lossy().to_string(),
            DeviceLibraryImportMode::Replace,
        )
        .expect("import replace");
        assert_eq!(replaced.len(), 1);

        create_device_template(CreateDeviceTemplateInput {
            name: "Local Device".to_string(),
            constraints: crate::device_profiles::DeviceResourceConstraints {
                mops_budget: 500.0,
                ram_budget_mb: 32.0,
                flash_budget_mb: 64.0,
                max_latency_ms: 20.0,
            },
            notes: None,
            tags: vec!["local".to_string()],
        })
        .expect("create local template");

        let merged = import_device_library(
            export_path.to_string_lossy().to_string(),
            DeviceLibraryImportMode::Merge,
        )
        .expect("import merge");

        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|t| t.name == "Exported Device"));
        assert!(merged.iter().any(|t| t.name == "Local Device"));

        let _ = fs::remove_dir_all(source_storage);
        let _ = fs::remove_dir_all(target_storage);
    }
}
