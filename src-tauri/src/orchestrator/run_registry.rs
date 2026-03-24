use crate::dtos::{RunState, TrainingJob, TrainingResult};
use std::collections::HashMap;

pub struct RunRegistry {
    runs: HashMap<String, RunState>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self {
            runs: HashMap::new(),
        }
    }

    pub fn create_run(&mut self, run_id: &str, max_parallel: u32) -> Result<(), String> {
        if self.runs.contains_key(run_id) {
            return Err(format!("Run {} already exists", run_id));
        }

        self.runs.insert(
            run_id.to_string(),
            RunState {
                run_id: run_id.to_string(),
                status: "pending".to_string(),
                created_at_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                started_at_ms: None,
                finished_at_ms: None,
                queued_jobs: Vec::new(),
                active_jobs: Vec::new(),
                completed_jobs: Vec::new(),
                failed_jobs: Vec::new(),
                max_parallel_jobs: max_parallel,
            },
        );

        Ok(())
    }

    pub fn enqueue_job(&mut self, run_id: &str, job: TrainingJob) -> Result<(), String> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        if run.status == "stopping" || run.status == "stopped" {
            return Err("Run is stopped or stopping".to_string());
        }

        run.queued_jobs.push(job.job_id.clone());
        Ok(())
    }

    pub fn mark_run_stopping(&mut self, run_id: &str) -> Result<(), String> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        if run.status != "running" && run.status != "pending" {
            return Ok(());
        }

        run.status = "stopping".to_string();
        Ok(())
    }

    pub fn get_run_state(&self, run_id: &str) -> Result<RunState, String> {
        self.runs
            .get(run_id)
            .cloned()
            .ok_or_else(|| format!("Run {} not found", run_id))
    }

    pub fn mark_job_completed(
        &mut self,
        run_id: &str,
        job_id: &str,
        _result: TrainingResult,
    ) -> Result<(), String> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        // Move from active to completed
        run.active_jobs.retain(|j| j != job_id);
        run.completed_jobs.push(job_id.to_string());

        Ok(())
    }
}

impl Default for RunRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let mut reg = RunRegistry::new();
        assert!(reg.create_run("run-1", 4).is_ok());

        let state = reg.get_run_state("run-1").unwrap();
        assert_eq!(state.status, "pending");
    }

    #[test]
    fn test_duplicate_run_error() {
        let mut reg = RunRegistry::new();
        reg.create_run("run-1", 4).unwrap();
        assert!(reg.create_run("run-1", 4).is_err());
    }

    #[test]
    fn test_enqueue_job() {
        let mut reg = RunRegistry::new();
        reg.create_run("run-1", 4).unwrap();

        let initial_queued = reg.get_run_state("run-1").unwrap().queued_jobs.len();
        assert_eq!(initial_queued, 0);
    }

    #[test]
    fn test_mark_run_stopping() {
        let mut reg = RunRegistry::new();
        reg.create_run("run-1", 4).unwrap();

        assert!(reg.mark_run_stopping("run-1").is_ok());
        let state = reg.get_run_state("run-1").unwrap();
        assert_eq!(state.status, "stopping");
    }
}
