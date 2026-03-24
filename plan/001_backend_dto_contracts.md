# Задача 001: Backend DTO Контракты

**Фаза**: 1 (Infrastructure)  
**Сложность**: Low  
**Время**: 4 часа  
**Зависимости**: None  
**Выполнит**: Backend разработчик

---

## Описание

Расширить `src-tauri/src/dtos.rs` с новыми структурами для profiling, Pareto, genealogy, device profiles и stopping criteria. Все DTO должны быть serializable/deserializable.

---

## Входные данные

- Текущий `src-tauri/src/dtos.rs` (уже содержит TrainingResult, EvaluationResult)
- План.md раздел 18-23 (описание структур)

---

## Пошаговое выполнение

### Шаг 1: Backup текущего файла
```bash
git checkout -b feat/dtos-expansion
```

### Шаг 2: Добавить TrainingProfiler DTO

В конец файла `dtos.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProfiler {
    pub train_start_ms: u64,
    pub first_batch_ms: u64,
    pub train_end_ms: u64,
    pub total_train_duration_ms: u64,
    
    pub val_start_ms: u64,
    pub val_end_ms: u64,
    pub val_duration_ms: u64,
    
    pub test_start_ms: u64,
    pub test_end_ms: u64,
    pub test_duration_ms: u64,
    
    pub peak_active_memory_mb: f32,
    pub peak_model_params_mb: f32,
    pub peak_gradient_mb: f32,
    pub peak_optim_state_mb: f32,
    pub peak_activation_mb: f32,
    
    pub samples_per_sec: f32,
    pub inference_msec_per_sample: f32,
    
    pub batch_count: u32,
    pub early_stop_epoch: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationProfilingStats {
    pub generation_number: u32,
    pub total_training_ms: u64,
    pub total_inference_ms: u64,
    pub avg_samples_per_sec: f32,
    pub peak_concurrent_vram_mb: f32,
    pub total_jobs_completed: u32,
    pub total_jobs_failed: u32,
}
```

### Шаг 3: Добавить GenomeObjectives (для Pareto)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeObjectives {
    pub genome_id: String,
    pub accuracy: f32,
    pub inference_latency_ms: f32,
    pub model_size_mb: f32,
    pub training_time_ms: u64,
    pub is_dominated: bool,
    pub domination_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParetoFront {
    pub generation: u32,
    pub total_genomes: u32,
    pub pareto_members: Vec<GenomeObjectives>,
    pub objectives_3d: Vec<(f32, f32, f32)>, // (accuracy, latency, size)
}
```

### Шаг 4: Добавить DeviceProfile

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeType {
    ARM,
    X86,
    GPU,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub device_id: String,
    pub device_name: String,
    pub compute_capability: ComputeType,
    pub ram_mb: u32,
    pub vram_mb: Option<u32>,
    pub inference_latency_budget_ms: f32,
    pub training_available: bool,
    pub power_budget_mw: Option<u32>,
    pub max_model_size_mb: Option<f32>,
    pub target_fps: Option<f32>,
}
```

### Шаг 5: Добавить Genealogy DTOs

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MutationType {
    Random,
    AddNode { node_type: String, source: String, target: String },
    RemoveNode { node_id: String },
    RemoveSubgraph { node_ids: Vec<String> },
    ParameterMutation { layer_id: String, param_name: String },
    ParameterScale { layer_id: String, scale_factor: f32 },
    Crossover { parent1: String, parent2: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeGeneology {
    pub genome_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,
    pub mutation_type: MutationType,
    pub mutation_params: serde_json::Value,
    pub fitness: f32,
    pub accuracy: f32,
    pub created_at_ms: u64,
}
```

### Шаг 6: Добавить Stopping Criteria DTOs

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoppingCriterion {
    GenerationLimit { max_generations: u32 },
    FitnessPlateau {
        patience_generations: u32,
        improvement_threshold: f32,
        monitor: String, // "best_fitness" | "pareto_coverage" | "population_avg"
    },
    TimeLimit { max_seconds: u32 },
    TargetAccuracy { threshold: f32 },
    ManualStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingPolicy {
    pub criteria: Vec<StoppingCriterion>,
    pub policy_type: String, // "any" | "all"
}
```

### Шаг 7: Обновить TrainingResult (добавить новые поля)

Найти существующий `TrainingResult` и добавить:

```rust
pub struct TrainingResult {
    // ... existing fields ...
    pub profiler: Option<TrainingProfiler>,           // NEW
    pub objectives: Option<GenomeObjectives>,        // NEW
    pub genealogy: Option<GenomeGeneology>,          // NEW
}
```

### Шаг 8: Обновить RustGenomeLibraryEntry

Найти в коде библиотеки и добавить:

```rust
pub struct RustGenomeLibraryEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub genome_json: String,
    pub created_at: String,
    pub is_pinned: bool,
    // NEW:
    pub is_hidden: bool,
    pub source_generation: Option<u32>,
    pub parent_genomes: Vec<String>,
    pub fitness_metrics: Option<GenomeObjectives>,
    pub profiler_data: Option<TrainingProfiler>,
    pub model_weights: Option<String>,
    pub device_profile_target: Option<DeviceProfile>,
}
```

### Шаг 9: Добавить Event DTOs

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "data")]
pub enum TrainingEvent {
    RunStarted { run_id: String },
    JobZeroCostScored {
        run_id: String,
        job_id: String,
        genome_id: String,
        proxy_score: f32,
        strategy_decision: String,
    },
    JobStarted { run_id: String, job_id: String, genome_id: String },
    JobProgress { run_id: String, job_id: String, epoch: u32, batch: u32, loss: f32 },
    JobFinished { run_id: String, job_id: String, genome_id: String, result: TrainingResult },
    GenerationParetoComputed { run_id: String, generation: u32, pareto_front: GenerationParetoFront },
}
```

---

## Критерии готовности

- ✅ Все новые DTO добавлены в `dtos.rs`
- ✅ Все структуры имеют `#[derive(Serialize, Deserialize)]`
- ✅ Код компилируется: `cargo build --release`
- ✅ Нет compiler warnings
- ✅ Файл готов к коммиту

---

## Тесты

### Unit тесты (добавить в конец dtos.rs):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_serialization() {
        let profiler = TrainingProfiler {
            train_start_ms: 1000,
            first_batch_ms: 1050,
            train_end_ms: 2000,
            total_train_duration_ms: 1000,
            val_start_ms: 2000,
            val_end_ms: 2100,
            val_duration_ms: 100,
            test_start_ms: 2100,
            test_end_ms: 2200,
            test_duration_ms: 100,
            peak_active_memory_mb: 1024.5,
            peak_model_params_mb: 512.0,
            peak_gradient_mb: 256.0,
            peak_optim_state_mb: 256.0,
            peak_activation_mb: 512.0,
            samples_per_sec: 100.0,
            inference_msec_per_sample: 10.0,
            batch_count: 100,
            early_stop_epoch: None,
        };
        
        let json = serde_json::to_string(&profiler).unwrap();
        let deserialized: TrainingProfiler = serde_json::from_str(&json).unwrap();
        assert_eq!(profiler.train_start_ms, deserialized.train_start_ms);
    }
    
    #[test]
    fn test_genome_objectives_serialization() {
        let obj = GenomeObjectives {
            genome_id: "test-1".to_string(),
            accuracy: 0.95,
            inference_latency_ms: 50.0,
            model_size_mb: 10.0,
            training_time_ms: 5000,
            is_dominated: false,
            domination_count: 0,
        };
        
        let json = serde_json::to_string(&obj).unwrap();
        let deserialized: GenomeObjectives = serde_json::from_str(&json).unwrap();
        assert_eq!(obj.accuracy, deserialized.accuracy);
    }
    
    #[test]
    fn test_mutation_type_serialization() {
        let mutations = vec![
            MutationType::Random,
            MutationType::AddNode {
                node_type: "Dense".to_string(),
                source: "n1".to_string(),
                target: "n2".to_string(),
            },
            MutationType::RemoveNode { node_id: "n1".to_string() },
            MutationType::Crossover {
                parent1: "g1".to_string(),
                parent2: "g2".to_string(),
            },
        ];
        
        for m in mutations {
            let json = serde_json::to_string(&m).unwrap();
            let deserialized: MutationType = serde_json::from_str(&json).unwrap();
            assert_eq!(m, deserialized);
        }
    }
    
    #[test]
    fn test_stopping_criteria_serialization() {
        let criteria = StoppingCriterion::FitnessPlateau {
            patience_generations: 5,
            improvement_threshold: 0.01,
            monitor: "best_fitness".to_string(),
        };
        
        let json = serde_json::to_string(&criteria).unwrap();
        let _deserialized: StoppingCriterion = serde_json::from_str(&json).unwrap();
    }
}
```

Запустить:
```bash
cargo test --lib dtos
```

---

## Вывод

- **Файл**: `src-tauri/src/dtos.rs` (расширение)
- **Строк кода**: ~150 новых LOC
- **Зависимость**: Все последующие задачи Phase 2

---

## Блокеры

- Если используется старая версия serde, обновить `Cargo.toml`

---

## Примечания

- Все DTO должны быть copy-friendly (derive Clone)
- Для Option fields использовать default serialization
- Не забыть про serde_json::Value для flexibility в mutation_params
