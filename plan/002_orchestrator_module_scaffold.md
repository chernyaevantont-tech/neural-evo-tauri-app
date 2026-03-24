# Задача 002: Orchestrator Module Scaffold

**Фаза**: 1 (Infrastructure)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 001  
**Выполнит**: Backend разработчик

---

## Описание

Создать модуль `orchestrator` в `src-tauri/src/` с каркасом для планировщика задач, управления памятью и координации параллельного обучения. Этот модуль используют все задачи Phase 2.

---

## Входные данные

- Новые DTO из Task 001 (dtos.rs)
- План.md раздел 6-10 (Scheduler, WGPU strategy, API contracts)

---

## Пошаговое выполнение

### Шаг 1: Создать структуру директорий

```bash
mkdir -p src-tauri/src/orchestrator
```

### Шаг 2: Создать `src-tauri/src/orchestrator/mod.rs`

```rust
pub mod scheduler;
pub mod memory_estimator;
pub mod run_registry;

pub use scheduler::Scheduler;
pub use memory_estimator::MemoryEstimator;
pub use run_registry::RunRegistry;

use crate::dtos::{TrainingJob, TrainingResult, RunState, StoppingPolicy};
use std::sync::{Arc, RwLock, Mutex};
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
        let mut registry = self.run_registry.write().unwrap();
        registry.create_run(&run_id, max_parallel_jobs)?;
        Ok(run_id)
    }

    /// Добавить job в очередь
    pub async fn enqueue_job(&self, run_id: &str, job: TrainingJob) -> Result<(), String> {
        let mut registry = self.run_registry.write().unwrap();
        registry.enqueue_job(run_id, job)?;
        
        // Запустить scheduler для дозапуска jobs
        self.scheduler.attempt_schedule(&run_id).await?;
        
        Ok(())
    }

    /// Остановить runs
    pub async fn stop_run(&self, run_id: &str) -> Result<(), String> {
        let mut registry = self.run_registry.write().unwrap();
        registry.mark_run_stopping(run_id)?;
        
        // Отменить активные jobs
        self.scheduler.cancel_active_jobs(run_id).await?;
        
        Ok(())
    }

    /// Получить статус runs
    pub fn get_run_status(&self, run_id: &str) -> Result<RunState, String> {
        let registry = self.run_registry.read().unwrap();
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
}
```

### Шаг 3: Создать `src-tauri/src/orchestrator/run_registry.rs`

```rust
use crate::dtos::{RunState, TrainingJob, TrainingResult};
use std::collections::HashMap;

pub struct RunRegistry {
    runs: HashMap<String, RunState>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self { runs: HashMap::new() }
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
                queued_jobs: vec![],
                active_jobs: vec![],
                completed_jobs: vec![],
                cancelled_jobs: vec![],
                reserved_vram_mb: 0,
                total_budget_mb: 8000, // TODO: make configurable
                max_parallel_jobs: max_parallel,
            },
        );

        Ok(())
    }

    pub fn enqueue_job(&mut self, run_id: &str, job: TrainingJob) -> Result<(), String> {
        let run = self.runs.get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        if run.status == "stopping" || run.status == "stopped" {
            return Err("Run is stopped or stopping".to_string());
        }

        run.queued_jobs.push(job);
        Ok(())
    }

    pub fn mark_run_stopping(&mut self, run_id: &str) -> Result<(), String> {
        let run = self.runs.get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        if run.status != "running" && run.status != "pending" {
            return Ok(());
        }

        run.status = "stopping".to_string();
        Ok(())
    }

    pub fn get_run_state(&self, run_id: &str) -> Result<RunState, String> {
        self.runs.get(run_id)
            .cloned()
            .ok_or_else(|| format!("Run {} not found", run_id))
    }

    pub fn mark_job_completed(&mut self, run_id: &str, job_id: &str, result: TrainingResult) -> Result<(), String> {
        let run = self.runs.get_mut(run_id)
            .ok_or_else(|| format!("Run {} not found", run_id))?;

        // Move from active to completed
        run.active_jobs.retain(|j| j != job_id);
        run.completed_jobs.push(job_id.to_string());

        Ok(())
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
}
```

### Шаг 4: Создать `src-tauri/src/orchestrator/memory_estimator.rs`

```rust
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtos::TrainingJob;

    #[test]
    fn test_vram_estimation() {
        let est = MemoryEstimator::new();
        let job = TrainingJob {
            job_id: "j1".to_string(),
            run_id: "r1".to_string(),
            genome_id: "g1".to_string(),
            genome_json: "{}".to_string(),
            epochs: 10,
            batch_size: 32,
            priority: 1,
            estimated_vram_mb: 1000,
            estimated_host_ram_mb: 2000,
            proxy_score: None,
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
}
```

### Шаг 5: Создать `src-tauri/src/orchestrator/scheduler.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct Scheduler {
    total_budget_mb: Arc<AtomicU64>,
    reserved_vram_mb: Arc<AtomicU64>,
    safety_margin_mb: u64,
}

impl Scheduler {
    pub fn new(total_budget_mb: u64, safety_margin_mb: u64) -> Self {
        Self {
            total_budget_mb: Arc::new(AtomicU64::new(total_budget_mb)),
            reserved_vram_mb: Arc::new(AtomicU64::new(0)),
            safety_margin_mb,
        }
    }

    pub fn available_vram_mb(&self) -> u64 {
        let total = self.total_budget_mb.load(Ordering::Relaxed);
        let reserved = self.reserved_vram_mb.load(Ordering::Relaxed);
        let usable = total.saturating_sub(self.safety_margin_mb);
        usable.saturating_sub(reserved)
    }

    pub fn try_reserve(&self, needed_mb: u64) -> bool {
        if self.available_vram_mb() >= needed_mb {
            let mut current = self.reserved_vram_mb.load(Ordering::Relaxed);
            loop {
                match self.reserved_vram_mb.compare_exchange(
                    current,
                    current + needed_mb,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(new_current) => current = new_current,
                }
            }
        }
        false
    }

    pub fn release(&self, freed_mb: u64) {
        self.reserved_vram_mb.fetch_sub(freed_mb, Ordering::Release);
    }

    pub async fn attempt_schedule(&self, _run_id: &str) -> Result<(), String> {
        // Placeholder: actual scheduling logic in later tasks
        Ok(())
    }

    pub async fn cancel_active_jobs(&self, _run_id: &str) -> Result<(), String> {
        // Placeholder: actual cancellation logic in later tasks
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let sched = Scheduler::new(8000, 500);
        assert_eq!(sched.available_vram_mb(), 7500);
    }

    #[test]
    fn test_reserve_and_release() {
        let sched = Scheduler::new(8000, 500);
        
        assert!(sched.try_reserve(1000));
        assert_eq!(sched.available_vram_mb(), 6500);
        
        sched.release(1000);
        assert_eq!(sched.available_vram_mb(), 7500);
    }

    #[test]
    fn test_reserve_overflow_protection() {
        let sched = Scheduler::new(8000, 500);
        assert!(sched.try_reserve(8000)); // Should fail (exceeds budget)
        assert!(!sched.try_reserve(8000));
    }
}
```

### Шаг 6: Добавить модуль в lib.rs

В `src-tauri/src/lib.rs`, добавить:

```rust
pub mod orchestrator;
```

### Шаг 7: Проверить компиляцию

```bash
cd src-tauri
cargo build
cargo test --lib orchestrator
```

---

## Критерии готовности

- ✅ Директория `src-tauri/src/orchestrator/` создана с 4 файлами
- ✅ `mod.rs` содержит TrainingOrchestrator с API
- ✅ `run_registry.rs` управляет состоянием runs и jobs
- ✅ `memory_estimator.rs` оценивает VRAM
- ✅ `scheduler.rs` управляет резервированием VRAM
- ✅ Все модули добавлены в `lib.rs`
- ✅ Компилируется без errors и warnings
- ✅ Unit тесты проходят
- ✅ Нет clippy рекомендаций

---

## Тесты

Все unit тесты уже вкомпилены выше. Запустить:

```bash
cargo test --lib orchestrator
cargo test --lib orchestrator::scheduler
cargo test --lib orchestrator::memory_estimator
cargo test --lib orchestrator::run_registry
```

---

## Вывод

- **Директория**: `src-tauri/src/orchestrator/`
- **Файлы**: `mod.rs`, `scheduler.rs`, `memory_estimator.rs`, `run_registry.rs`
- **Строк кода**: ~300 новых LOC
- **Зависимость**: Все задачи Phase 2 используют этот модуль

---

## Примечания

- Scheduler использует atomics для thread-safety
- RunRegistry использует RwLock для конкурентного доступа
- Все тестовые значения (8000MB, 500MB margin) подлежат уточнению позже
- API placeholder для cancel_active_jobs и attempt_schedule - реализуются в Phase 2 tasks
- Scheduler поддерживает только FIFO без приоритета (добавится позже если нужно)
