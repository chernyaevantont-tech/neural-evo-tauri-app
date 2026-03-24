use crate::dtos::TrainingJob;

pub struct MemoryEstimator;

impl MemoryEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Оценить VRAM необходимую для одного job
    pub fn estimate_vram_for_job(&self, job: &TrainingJob) -> u64 {
        // Консервативная оценка:
        // M_job = M_params + M_grads + M_optim + M_activations + M_batch + M_workspace
        // Для Adam (fp32): M_params_grads_optim ~= 4 * M_params

        let base_estimate = job.estimated_vram_mb;
        let safety_factor = 1.5;

        (base_estimate as f64 * safety_factor) as u64
    }

    /// Estimate max parallel jobs that fit in VRAM
    pub fn estimate_max_parallel_fit(
        &self,
        available_vram_mb: u64,
        job_vram_mb: u64,
        safety_margin_mb: u64,
    ) -> u32 {
        let usable = available_vram_mb.saturating_sub(safety_margin_mb);
        (usable / job_vram_mb) as u32
    }

    /// Estimate total job count fit
    pub fn estimate_batch_fit(&self, total_vram_mb: u64, job_vram_mb: u64) -> u32 {
        (total_vram_mb / job_vram_mb) as u32
    }
}

impl Default for MemoryEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_estimation() {
        let est = MemoryEstimator::new();
        let job = TrainingJob {
            job_id: "j1".to_string(),
            run_id: "r1".to_string(),
            genome_id: "g1".to_string(),
            genome_json: "{}".to_string(),
            training_params: serde_json::json!({"epochs": 10}),
            estimated_vram_mb: 1000,
            dataset_name: "mnist".to_string(),
            priority: 0,
            created_at_ms: 0,
            proxy_decision: None,
        };

        let vram = est.estimate_vram_for_job(&job);
        assert!(vram > 1000); // Should apply safety factor
    }

    #[test]
    fn test_parallel_fit_calculation() {
        let est = MemoryEstimator::new();
        let available = 8000;
        let job_vram = 1000;
        let safety_margin = 500;

        let fit = est.estimate_max_parallel_fit(available, job_vram, safety_margin);
        assert!(fit > 0 && fit <= 10);
    }

    #[test]
    fn test_batch_fit_calculation() {
        let est = MemoryEstimator::new();
        let total = 8000;
        let job_vram = 1000;

        let fit = est.estimate_batch_fit(total, job_vram);
        assert_eq!(fit, 8);
    }
}
