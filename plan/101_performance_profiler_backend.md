# Задача 101: Performance Profiler Backend

**Фаза**: 2 (Core Features - Metrics)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001 (dtos)  
**Выполнит**: Backend разработчик (Rust/Burn)

---

## Описание

Реализовать `TrainingProfiler` в backend: инструментировать обучение для сбора timing (training/val/test), memory peaks, throughput. Интегрировать в training loop (entities.rs, run_eval_pass).

---

## Входные данные

- `src-tauri/src/dtos.rs` (TrainingProfiler DTO из Task 001)
- `src-tauri/src/entities.rs` (training loop: run_eval_pass, run_validation_pass)
- План.md раздел 18.2 (instrumentation points)

---

## Пошаговое выполнение

### Шаг 1: Создать модуль profiler

Создать `src-tauri/src/profiler.rs`:

```rust
use std::time::Instant;
use std::sync::{Arc, Mutex};

pub struct ProfilerCollector {
    train_start: Option<Instant>,
    first_batch: Option<Instant>,
    train_end: Option<Instant>,
    
    val_start: Option<Instant>,
    val_end: Option<Instant>,
    
    test_start: Option<Instant>,
    test_end: Option<Instant>,
    
    peak_memory_mb: Arc<Mutex<f32>>,
    batch_count: u32,
}

impl ProfilerCollector {
    pub fn new() -> Self {
        Self {
            train_start: None,
            first_batch: None,
            train_end: None,
            val_start: None,
            val_end: None,
            test_start: None,
            test_end: None,
            peak_memory_mb: Arc::new(Mutex::new(0.0)),
            batch_count: 0,
        }
    }

    pub fn mark_train_start(&mut self) {
        self.train_start = Some(Instant::now());
    }

    pub fn mark_first_batch(&mut self) {
        if self.first_batch.is_none() {
            self.first_batch = Some(Instant::now());
        }
    }

    pub fn mark_train_end(&mut self) {
        self.train_end = Some(Instant::now());
    }

    pub fn mark_val_start(&mut self) {
        self.val_start = Some(Instant::now());
    }

    pub fn mark_val_end(&mut self) {
        self.val_end = Some(Instant::now());
    }

    pub fn mark_test_start(&mut self) {
        self.test_start = Some(Instant::now());
    }

    pub fn mark_test_end(&mut self) {
        self.test_end = Some(Instant::now());
    }

    pub fn record_batch(&mut self) {
        self.batch_count += 1;
    }

    pub fn update_peak_memory(&self, current_mb: f32) {
        let mut peak = self.peak_memory_mb.lock().unwrap();
        if current_mb > *peak {
            *peak = current_mb;
        }
    }

    pub fn finalize(&self) -> crate::dtos::TrainingProfiler {
        let train_start_ms = self.train_start.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0);
        let first_batch_ms = self.first_batch
            .map(|i| {
                let elapsed = i.elapsed().as_millis() as u64;
                (self.train_start.unwrap().elapsed().as_millis() as u64) - elapsed
            })
            .unwrap_or(0);
        let total_train_duration_ms = self.train_end
            .map(|i| {
                let now = i.elapsed().as_millis() as u64;
                train_start_ms - now
            })
            .unwrap_or(0);

        let val_duration_ms = if let (Some(start), Some(end)) = (self.val_start, self.val_end) {
            (start.elapsed().as_millis() as u64) - (end.elapsed().as_millis() as u64)
        } else {
            0
        };

        let test_duration_ms = if let (Some(start), Some(end)) = (self.test_start, self.test_end) {
            (start.elapsed().as_millis() as u64) - (end.elapsed().as_millis() as u64)
        } else {
            0
        };

        let peak_memory = *self.peak_memory_mb.lock().unwrap();

        crate::dtos::TrainingProfiler {
            train_start_ms,
            first_batch_ms,
            train_end_ms: (self.train_end.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0)),
            total_train_duration_ms,
            val_start_ms: (self.val_start.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0)),
            val_end_ms: (self.val_end.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0)),
            val_duration_ms,
            test_start_ms: (self.test_start.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0)),
            test_end_ms: (self.test_end.map(|i| i.elapsed().as_millis() as u64).unwrap_or(0)),
            test_duration_ms,
            peak_active_memory_mb: peak_memory,
            peak_model_params_mb: 0.0, // TODO: compute from model
            peak_gradient_mb: 0.0,     // TODO: compute from gradients
            peak_optim_state_mb: 0.0,  // TODO: compute from optimizer state
            peak_activation_mb: 0.0,   // TODO: compute from activations
            samples_per_sec: if total_train_duration_ms > 0 {
                (self.batch_count as f32 * 32.0) / (total_train_duration_ms as f32 / 1000.0)
            } else {
                0.0
            },
            inference_msec_per_sample: if self.batch_count > 0 {
                (val_duration_ms as f32) / (self.batch_count as f32 * 32.0)
            } else {
                0.0
            },
            batch_count: self.batch_count,
            early_stop_epoch: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = ProfilerCollector::new();
        assert_eq!(profiler.batch_count, 0);
    }

    #[test]
    fn test_profiler_finalization() {
        let mut profiler = ProfilerCollector::new();
        profiler.mark_train_start();
        profiler.mark_first_batch();
        std::thread::sleep(std::time::Duration::from_millis(100));
        profiler.mark_train_end();
        
        let result = profiler.finalize();
        assert!(result.total_train_duration_ms > 0);
    }
}
```

### Шаг 2: Добавить profiler в lib.rs

```rust
pub mod profiler;
use profiler::ProfilerCollector;
```

### Шаг 3: Интегрировать в training loop (entities.rs)

В функции `run_eval_pass`, найти начало training loop и добавить профилер:

```rust
pub async fn run_eval_pass(
    model: &mut GraphModel,
    train_batch: Tensor,
    train_targets: Tensor,
    config: &TrainConfig,
    profiler: &mut Option<ProfilerCollector>,
) -> f32 {
    if let Some(p) = profiler {
        p.mark_train_start();
    }

    let mut loss_sum = 0.0;
    
    for epoch in 0..config.epochs {
        // Training loop
        for batch_idx in 0..config.batch_count {
            if batch_idx == 0 && epoch == 0 {
                if let Some(p) = profiler {
                    p.mark_first_batch();
                }
            }

            // Forward pass
            let output = model.forward(&train_batch);
            let loss = compute_loss(&output, &train_targets);
            loss_sum += loss.into_scalar() as f32;

            // Backward pass
            loss.backward();
            
            // Record batch
            if let Some(p) = profiler {
                p.record_batch();
            }

            // Update weights
            model.optimizer_step();
        }
    }

    if let Some(p) = profiler {
        p.mark_train_end();
    }
    
    loss_sum / (config.epochs as f32)
}
```

### Шаг 4: Интегрировать в validation pass

```rust
pub async fn run_validation_pass(
    model: &GraphModel,
    val_batch: Tensor,
    val_targets: Tensor,
    profiler: &mut Option<ProfilerCollector>,
) -> (f32, f32) { // (loss, accuracy)
    if let Some(p) = profiler {
        p.mark_val_start();
    }

    let output = model.forward(&val_batch);
    let loss = compute_loss(&output, &val_targets);
    let accuracy = compute_accuracy(&output, &val_targets);

    if let Some(p) = profiler {
        p.mark_val_end();
    }

    (loss.into_scalar() as f32, accuracy)
}
```

### Шаг 5: Интегрировать в test pass

```rust
pub async fn run_test_pass(
    model: &GraphModel,
    test_batch: Tensor,
    test_targets: Tensor,
    profiler: &mut Option<ProfilerCollector>,
) -> f32 { // accuracy
    if let Some(p) = profiler {
        p.mark_test_start();
    }

    let output = model.forward(&test_batch);
    let accuracy = compute_accuracy(&output, &test_targets);

    if let Some(p) = profiler {
        p.mark_test_end();
    }

    accuracy
}
```

### Шаг 6: Обновить TrainingResult с профилером

В функции `evaluate_genome` или где создается TrainingResult:

```rust
let mut profiler = Some(ProfilerCollector::new());

// ... training code ...

let result = TrainingResult {
    genome_id: genome.id.clone(),
    loss,
    accuracy,
    profiler: profiler.map(|p| p.finalize()),
    // ... other fields ...
};
```

### Шаг 7: Проверить компиляцию

```bash
cd src-tauri
cargo build --release
cargo test --lib profiler
```

---

## Критерии готовности

- ✅ Модуль `src-tauri/src/profiler.rs` создан
- ✅ `ProfilerCollector` реализован с методами marking и finalization
- ✅ Интегрировано в `run_eval_pass`, `run_validation_pass`, `run_test_pass`
- ✅ `TrainingResult.profiler` заполняется финальными данными
- ✅ Компилируется без ошибок
- ✅ Unit тесты проходят

---

## Тесты

```rust
#[cfg(test)]
mod profiler_tests {
    use super::*;

    #[test]
    fn test_profiler_timing_accuracy() {
        let mut profiler = ProfilerCollector::new();
        profiler.mark_train_start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        profiler.mark_train_end();
        
        let result = profiler.finalize();
        assert!(result.total_train_duration_ms >= 200);
    }

    #[test]
    fn test_batch_counting() {
        let mut profiler = ProfilerCollector::new();
        for _ in 0..50 {
            profiler.record_batch();
        }
        let result = profiler.finalize();
        assert_eq!(result.batch_count, 50);
    }

    #[test]
    fn test_throughput_calculation() {
        let mut profiler = ProfilerCollector::new();
        profiler.mark_train_start();
        for _ in 0..100 {
            profiler.record_batch();
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
        profiler.mark_train_end();
        
        let result = profiler.finalize();
        assert!(result.samples_per_sec > 0.0);
    }
}
```

---

## Вывод

- **Файл**: `src-tauri/src/profiler.rs` (новый)
- **Изменения**: `entities.rs` (3 функции, ~40 LOC добавлено)
- **Строк кода**: ~200 новых LOC
- **Интеграция**: Используется в Task 104 (frontend display), Task 109 (hidden library storage)

---

## Примечания

- Memory tracking (peak_*_mb) TBD: требует более глубокой интеграции с Burn allocator
- Timing может быть неточным на очень быстрых операциях (< 1ms), но достаточно для профилирования обучения
- Профилер thread-safe благодаря Arc<Mutex<>>
- Throughput вычисляется как (batches × batch_size) / (duration_seconds), где batch_size hardcoded=32 (TODO: parametrize)
