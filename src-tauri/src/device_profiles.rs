use crate::dtos::GenomeObjectives;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceResourceConstraints {
    pub mops_budget: f32,
    pub ram_budget_mb: f32,
    pub flash_budget_mb: f32,
    pub max_latency_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceValidationResult {
    pub is_feasible: bool,
    pub violation_score: f32,
    pub mops_ratio: f32,
    pub ram_ratio: f32,
    pub flash_ratio: f32,
    pub latency_ratio: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceProfileDto {
    pub profile_id: String,
    pub profile_name: String,
    pub constraints: DeviceResourceConstraints,
}

pub fn validate_constraints(constraints: &DeviceResourceConstraints) -> Result<(), String> {
    if constraints.mops_budget <= 0.0 {
        return Err("mops_budget must be > 0".to_string());
    }
    if constraints.ram_budget_mb <= 0.0 {
        return Err("ram_budget_mb must be > 0".to_string());
    }
    if constraints.flash_budget_mb <= 0.0 {
        return Err("flash_budget_mb must be > 0".to_string());
    }
    if constraints.max_latency_ms <= 0.0 {
        return Err("max_latency_ms must be > 0".to_string());
    }

    Ok(())
}

fn safe_ratio(usage: f32, budget: f32) -> f32 {
    if budget <= 0.0 {
        return f32::INFINITY;
    }
    usage / budget
}

fn estimate_mops(objectives: &GenomeObjectives) -> f32 {
    // Until dedicated MACs/FLOPs stats are added to GenomeObjectives,
    // use a coarse proxy from latency and model size to keep constraint
    // validation monotonic and deterministic.
    (objectives.inference_latency_ms.max(0.0) * 1.5) + (objectives.model_size_mb.max(0.0) * 12.0)
}

fn estimate_ram_mb(objectives: &GenomeObjectives) -> f32 {
    // Runtime RAM is usually > model size due to activations/intermediate buffers.
    objectives.model_size_mb.max(0.0) * 2.5
}

fn estimate_flash_mb(objectives: &GenomeObjectives) -> f32 {
    // Persisted model footprint approximation.
    objectives.model_size_mb.max(0.0)
}

pub fn built_in_profiles() -> Vec<DeviceProfileDto> {
    vec![
        DeviceProfileDto {
            profile_id: "embedded-mcu".to_string(),
            profile_name: "Embedded MCU".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 120.0,
                ram_budget_mb: 0.5,
                flash_budget_mb: 2.0,
                max_latency_ms: 80.0,
            },
        },
        DeviceProfileDto {
            profile_id: "edge-tiny".to_string(),
            profile_name: "Edge Tiny".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 800.0,
                ram_budget_mb: 8.0,
                flash_budget_mb: 32.0,
                max_latency_ms: 50.0,
            },
        },
        DeviceProfileDto {
            profile_id: "mobile-low-end".to_string(),
            profile_name: "Mobile Low-End".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 3000.0,
                ram_budget_mb: 128.0,
                flash_budget_mb: 256.0,
                max_latency_ms: 45.0,
            },
        },
        DeviceProfileDto {
            profile_id: "mobile-mid-range".to_string(),
            profile_name: "Mobile Mid-Range".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 8000.0,
                ram_budget_mb: 256.0,
                flash_budget_mb: 512.0,
                max_latency_ms: 30.0,
            },
        },
        DeviceProfileDto {
            profile_id: "laptop-cpu".to_string(),
            profile_name: "Laptop CPU".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 25000.0,
                ram_budget_mb: 2048.0,
                flash_budget_mb: 2048.0,
                max_latency_ms: 25.0,
            },
        },
        DeviceProfileDto {
            profile_id: "laptop-igpu".to_string(),
            profile_name: "Laptop iGPU".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 70000.0,
                ram_budget_mb: 4096.0,
                flash_budget_mb: 4096.0,
                max_latency_ms: 15.0,
            },
        },
        DeviceProfileDto {
            profile_id: "workstation".to_string(),
            profile_name: "Workstation".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 220000.0,
                ram_budget_mb: 16384.0,
                flash_budget_mb: 8192.0,
                max_latency_ms: 8.0,
            },
        },
        DeviceProfileDto {
            profile_id: "cloud-t4".to_string(),
            profile_name: "Cloud T4".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 350000.0,
                ram_budget_mb: 16384.0,
                flash_budget_mb: 16384.0,
                max_latency_ms: 6.0,
            },
        },
        DeviceProfileDto {
            profile_id: "cloud-a100".to_string(),
            profile_name: "Cloud A100".to_string(),
            constraints: DeviceResourceConstraints {
                mops_budget: 1_200_000.0,
                ram_budget_mb: 81920.0,
                flash_budget_mb: 65536.0,
                max_latency_ms: 2.0,
            },
        },
    ]
}

pub fn validate_genome_for_device(
    objectives: &GenomeObjectives,
    constraints: &DeviceResourceConstraints,
) -> DeviceValidationResult {
    let mops_ratio = safe_ratio(estimate_mops(objectives), constraints.mops_budget);
    let ram_ratio = safe_ratio(estimate_ram_mb(objectives), constraints.ram_budget_mb);
    let flash_ratio = safe_ratio(estimate_flash_mb(objectives), constraints.flash_budget_mb);
    let latency_ratio = safe_ratio(
        objectives.inference_latency_ms.max(0.0),
        constraints.max_latency_ms,
    );

    let mops_excess = (mops_ratio - 1.0).max(0.0);
    let ram_excess = (ram_ratio - 1.0).max(0.0);
    let flash_excess = (flash_ratio - 1.0).max(0.0);
    let latency_excess = (latency_ratio - 1.0).max(0.0);

    let violation_score = (latency_excess * latency_excess * 0.35)
        + (mops_excess * mops_excess * 0.30)
        + (ram_excess * ram_excess * 0.20)
        + (flash_excess * flash_excess * 0.15);

    let is_feasible = mops_ratio <= 1.0 && ram_ratio <= 1.0 && flash_ratio <= 1.0 && latency_ratio <= 1.0;

    DeviceValidationResult {
        is_feasible,
        violation_score,
        mops_ratio,
        ram_ratio,
        flash_ratio,
        latency_ratio,
    }
}

pub fn apply_device_penalty(base_fitness: f32, violation_score: f32, alpha: f32) -> f32 {
    base_fitness - alpha.max(0.0) * violation_score.max(0.0)
}

pub fn score_fitness_with_device_constraints(
    base_fitness: f32,
    objectives: &GenomeObjectives,
    constraints: &DeviceResourceConstraints,
    alpha: f32,
) -> (f32, DeviceValidationResult) {
    let validation = validate_genome_for_device(objectives, constraints);
    let adjusted = apply_device_penalty(base_fitness, validation.violation_score, alpha);
    (adjusted, validation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objectives(latency_ms: f32, model_size_mb: f32) -> GenomeObjectives {
        GenomeObjectives {
            genome_id: "g1".to_string(),
            accuracy: 0.93,
            inference_latency_ms: latency_ms,
            model_size_mb,
            training_time_ms: 1200,
            is_dominated: false,
            domination_count: 0,
        }
    }

    #[test]
    fn profile_list_contains_nine_built_ins() {
        let profiles = built_in_profiles();
        assert_eq!(profiles.len(), 9);
    }

    #[test]
    fn feasible_genome_has_zero_violation_score() {
        let constraints = DeviceResourceConstraints {
            mops_budget: 1000.0,
            ram_budget_mb: 64.0,
            flash_budget_mb: 64.0,
            max_latency_ms: 20.0,
        };
        let genome = objectives(10.0, 8.0);

        let validation = validate_genome_for_device(&genome, &constraints);
        assert!(validation.is_feasible);
        assert_eq!(validation.violation_score, 0.0);
    }

    #[test]
    fn individual_resource_excess_yields_positive_penalty() {
        let constraints = DeviceResourceConstraints {
            mops_budget: 100.0,
            ram_budget_mb: 16.0,
            flash_budget_mb: 8.0,
            max_latency_ms: 5.0,
        };

        let high_latency = objectives(20.0, 2.0);
        let high_flash = objectives(2.0, 16.0);
        let high_ram = objectives(2.0, 20.0);
        let high_mops = objectives(50.0, 10.0);

        assert!(validate_genome_for_device(&high_latency, &constraints).violation_score > 0.0);
        assert!(validate_genome_for_device(&high_flash, &constraints).violation_score > 0.0);
        assert!(validate_genome_for_device(&high_ram, &constraints).violation_score > 0.0);
        assert!(validate_genome_for_device(&high_mops, &constraints).violation_score > 0.0);
    }

    #[test]
    fn penalty_grows_quadratically_on_same_dimension() {
        let constraints = DeviceResourceConstraints {
            mops_budget: 10_000.0,
            ram_budget_mb: 10_000.0,
            flash_budget_mb: 10_000.0,
            max_latency_ms: 10.0,
        };
        let mild = objectives(12.0, 1.0);
        let severe = objectives(16.0, 1.0);

        let p1 = validate_genome_for_device(&mild, &constraints).violation_score;
        let p2 = validate_genome_for_device(&severe, &constraints).violation_score;

        assert!(p2 > p1);
        let ratio = p2 / p1;
        assert!(ratio > 8.5 && ratio < 9.5);
    }

    #[test]
    fn serde_roundtrip_for_api_payloads() {
        let payload = DeviceResourceConstraints {
            mops_budget: 1000.0,
            ram_budget_mb: 128.0,
            flash_budget_mb: 256.0,
            max_latency_ms: 30.0,
        };

        let json = serde_json::to_string(&payload).expect("serialize constraints");
        let de: DeviceResourceConstraints =
            serde_json::from_str(&json).expect("deserialize constraints");
        assert_eq!(payload.max_latency_ms, de.max_latency_ms);
    }
}