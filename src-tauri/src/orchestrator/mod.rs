pub mod scheduler;
pub mod memory_estimator;
pub mod run_registry;

pub use scheduler::Scheduler;
pub use memory_estimator::MemoryEstimator;
pub use run_registry::RunRegistry;

use crate::dtos::{TrainingJob, RunState, StoppingPolicy};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Центральный оркестратор для управления параллельным обучением
pub struct TrainingOrchestrator {
    pub run_registry: Arc<RwLock<RunRegistry>>,
    pub scheduler: Arc<Scheduler>,
    pub memory_estimator: Arc<MemoryEstimator>,
    pub stopping_policy: Arc<RwLock<StoppingPolicy>>,
}

impl TrainingOrchestrator {
    pub fn new(memory_budget_mb: u64, safety_margin_mb: u64) -> Self {
        Self {
            run_registry: Arc::new(RwLock::new(RunRegistry::new())),
            scheduler: Arc::new(Scheduler::new(memory_budget_mb, safety_margin_mb)),
            memory_estimator: Arc::new(MemoryEstimator::new()),
            stopping_policy: Arc::new(RwLock::new(StoppingPolicy {
                criteria: vec![],
                policy_type: "any".to_string(),
            })),
        }
    }

    /// Создать новый training run
    pub async fn start_training_run(&self, max_parallel_jobs: u32) -> Result<String, String> {
        let run_id = Uuid::new_v4().to_string();
        let mut registry = self.run_registry.write().map_err(|e| format!("Lock error: {}", e))?;
        registry.create_run(&run_id, max_parallel_jobs)?;
        Ok(run_id)
    }

    /// Добавить job в очередь
    pub async fn enqueue_job(&self, run_id: &str, job: TrainingJob) -> Result<(), String> {
        let mut registry = self.run_registry.write().map_err(|e| format!("Lock error: {}", e))?;
        registry.enqueue_job(run_id, job)?;

        // Запустить scheduler для дозапуска jobs
        self.scheduler.attempt_schedule(run_id).await?;

        Ok(())
    }

    /// Остановить runs
    pub async fn stop_run(&self, run_id: &str) -> Result<(), String> {
        let mut registry = self.run_registry.write().map_err(|e| format!("Lock error: {}", e))?;
        registry.mark_run_stopping(run_id)?;

        // Отменить активные jobs
        self.scheduler.cancel_active_jobs(run_id).await?;

        Ok(())
    }

    /// Получить статус runs
    pub fn get_run_status(&self, run_id: &str) -> Result<RunState, String> {
        let registry = self
            .run_registry
            .read()
            .map_err(|e| format!("Lock error: {}", e))?;
        registry.get_run_state(run_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orch = TrainingOrchestrator::new(8000, 500);
        assert!(orch.scheduler.available_vram_mb() > 0);
    }

    #[tokio::test]
    async fn test_start_training_run() {
        let orch = TrainingOrchestrator::new(8000, 500);
        let run_id = orch.start_training_run(4).await.unwrap();
        assert!(!run_id.is_empty());

        let status = orch.get_run_status(&run_id).unwrap();
        assert_eq!(status.status, "pending");
    }

    #[tokio::test]
    async fn test_stop_run() {
        let orch = TrainingOrchestrator::new(8000, 500);
        let run_id = orch.start_training_run(4).await.unwrap();

        assert!(orch.stop_run(&run_id).await.is_ok());

        let status = orch.get_run_status(&run_id).unwrap();
        assert_eq!(status.status, "stopping");
    }

    #[tokio::test]
    async fn test_multiple_runs() {
        let orch = TrainingOrchestrator::new(8000, 500);

        let run1 = orch.start_training_run(4).await.unwrap();
        let run2 = orch.start_training_run(4).await.unwrap();

        assert_ne!(run1, run2);

        let status1 = orch.get_run_status(&run1).unwrap();
        let status2 = orch.get_run_status(&run2).unwrap();

        assert_eq!(status1.status, "pending");
        assert_eq!(status2.status, "pending");
    }
}
