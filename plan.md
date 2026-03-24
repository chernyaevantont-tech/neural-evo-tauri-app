# Plan: Параллельное обучение геномов на одной GPU (WGPU only)

## 1. Цель и рамки

Цель: полностью перейти на параллельную модель обучения геномов в backend.

Ключевое требование:
- Последовательный режим обучения не поддерживается.
- Запуск идет только через очередь задач + пул воркеров.
- Новая задача стартует только если после резервирования памяти остается безопасный VRAM-бюджет.
- Если не помещается даже 1 задача, запуск эволюции блокируется до старта с понятной ошибкой preflight.

Что должно быть в итоге:
- Одновременно обучается несколько геномов в рамках одного WGPU-устройства.
- Планировщик динамически дозапускает задачи из очереди, как только освобождается бюджет памяти.
- Отмена обучения и закрытие приложения гарантированно останавливают прием новых задач, отменяют активные и освобождают ресурсы.
- Zero-Cost Proxies могут включаться как этап предварительного скоринга и влияют на решение: skip / partial_train / full_train.

## 2. Текущее состояние (as-is)

Backend сейчас:
- Оценка геномов делается в цикле по одному геному.
- Есть проверка отмены через global session counter.
- Обучение уходит в spawn_blocking, но await внутри последовательного цикла, поэтому фактической параллельности нет.

Frontend сейчас:
- UI событийно ориентирован на "один активный геном".
- stop отправляет backend-команду, но close окна не делает coordinated shutdown оркестратора.

## 3. Целевая архитектура (to-be)

Новая схема:
1. Tauri command start_parallel_run создает TrainingRun (run_id) и Job Queue.
2. Геномы добавляются в очередь через enqueue_genome_job (по одному или мини-пачками, но это не обязательное условие).
3. Backend выполняет preflight памяти до старта run и runtime admission при каждом запуске job.
4. Scheduler запускает worker-задачи строго по admission control:
   - проверяет свободный budget;
   - резервирует memory токены;
   - стартует job;
   - по завершении освобождает токены.
5. Как только любой worker завершил job, scheduler немедленно пытается взять следующую задачу из очереди.
6. События прогресса/результатов эмитятся по job_id/genome_id.
7. На stop/close orchestrator переходит в shutdown state и корректно завершает run.

Принцип: queue-first, а не batch-first.
- Передача всех геномов одним массивом допустима только как удобный клиентский шорткат.
- Внутренняя модель выполнения всегда одна: очередь и дозапуск следующего job при освобождении ресурса.

### 3.1 Режим с Zero-Cost Proxies

Если Zero-Cost Proxies включены:
1. Геном сначала проходит proxy-оценку (SynFlow/другой proxy).
2. По policy принимается решение:
  - skip: не отправлять в training queue, финальный fitness строится из proxy-метрики.
  - partial_train: добавить в очередь с уменьшенным epochs.
  - full_train: добавить в очередь с полным budget epochs.
3. Решение и score логируются и эмитятся в UI до старта реального обучения.

Если Zero-Cost Proxies выключены:
- Каждый геном сразу попадает в training queue.

## 4. Новые доменные сущности

### 4.1 TrainingJob
- job_id: UUID
- run_id: UUID
- genome_id: string
- genome_json: string
- epochs: usize
- batch_size: usize
- priority: u8
- estimated_vram_mb: u64
- estimated_host_ram_mb: u64
- proxy_score: Option<f32>
- proxy_decision: Option<skip | partial_train | full_train>

### 4.2 TrainingResult
- job_id
- genome_id
- status: success | cancelled | rejected_oom | failed
- loss: f32
- accuracy: f32
- started_at / finished_at
- duration_ms
- peak_estimated_vram_mb
- error_message: Option<String>
- zero_cost_used: bool
- proxy_score: Option<f32>
- strategy_decision: Option<skip | partial_train | full_train>

### 4.3 RunState
- run_id
- status: pending | running | stopping | stopped | finished | failed
- queued_jobs
- active_jobs
- completed_jobs
- cancelled_jobs
- reserved_vram_mb
- total_budget_mb

### 4.4 MemoryBudget
- total_budget_mb
- safety_margin_mb
- max_parallel_jobs_hard_cap
- reserved_now_mb

## 5. Preflight-проверка памяти (до старта)

### 5.1 Вход
- список геномов
- batch_size
- epochs per genome
- тип устройства (APU/discrete/manual)
- safety_margin_mb
- коэффициент запаса estimator_safety_factor
- флаг use_zero_cost_proxies и policy thresholds

### 5.2 Оценка памяти на job
Консервативная оценка:

M_job = M_params + M_grads + M_optim + M_activations + M_batch + M_workspace

Для Adam (fp32) брать:
- M_params_grads_optim ~= 4 * M_params

Активации:
- суммировать по слоям тензоры прямого/обратного прохода
- применять safety factor (например 1.2-1.5)

Итог:
- estimated_vram_mb_per_genome
- worst_case_parallel_fit_count

### 5.3 Логика preflight
- Если estimated_vram_mb_min > available_budget_mb: reject start.
- Иначе рассчитать recommended_parallelism >= 1.
- В ответ вернуть матрицу: genome_id -> estimate, общий budget, max concurrency.

Если включен Zero-Cost режим:
- preflight дополнительно должен считать два сценария:
  - raw (без proxy-фильтра)
  - filtered (после решений skip/partial/full)
- Для filtered сценария считать expected queue load и expected parallel throughput.

Важно:
- Никакого fallback в последовательный режим.
- Если пользователь выставил max_parallel_jobs=8, а preflight дает 3, система запускает максимум 3 параллельно, остальные стоят в очереди.

## 6. Планировщик: очередь + пул воркеров + admission control

### 6.1 Основные правила
- Есть одна FIFO очередь pending jobs (возможно с приоритетом в будущем).
- Есть active set воркеров.
- На каждом tick scheduler пытается стартовать jobs:
  - while active < max_parallel_jobs
  - и while next_job.estimated_vram_mb <= free_budget_mb
  - стартует next_job.

При включенном Zero-Cost:
- skip-геномы не резервируют VRAM и сразу переводятся в completed (proxy_based).
- partial_train/full_train попадают в queue как обычные training jobs.

### 6.2 Резервирование памяти
- До старта job резервирует estimated_vram_mb.
- После finish/cancel/fail reservation снимается.
- Резервация atomic под mutex/RwLock.

### 6.3 Контроль конкурентности
Финальное число активных job:
- min(
  user_max_parallel_jobs,
  preflight_recommended_parallelism,
  runtime_fit_by_reserved_budget
)

### 6.4 Runtime oversubscription guard
- Если во время run появляется признак нехватки памяти (OOM/allocator failure),
  - текущую job помечать failed/rejected_oom,
  - снижать dynamic concurrency лимит (например -1),
  - переоценивать оставшуюся очередь.

### 6.5 Приоритизация очереди с proxy-данными
- Базовый режим: FIFO.
- Опциональный режим: priority queue по proxy_score и proxy_decision.
- Рекомендуемая политика:
  - full_train с высоким proxy_score могут идти раньше (ускоряет получение сильных кандидатов).
  - partial_train с низким score могут откладываться.
  - fairness guard: starvation запрещен, у каждой job есть max_wait.

## 7. WGPU/Burn стратегия выполнения

### 7.1 Device context
- Использовать единый WgpuDevice для run (или централизованный device provider).
- Не создавать новый тяжелый контекст на каждый mini-step.

### 7.2 Worker модель
- Каждый worker исполняет один TrainingJob до terminal state.
- Training/validation/test внутри job остаются синхронными в контексте worker.
- Параллелизм достигается на уровне нескольких worker jobs одновременно.

### 7.3 Ограничение нагрузки
- Межбатчевое cooperative yielding оставить.
- Добавить configurable throttle (например sleep 0-2 ms), чтобы GPU не фризила UI-пайплайн на display GPU.

## 8. Новый контракт backend API

### 8.1 Команды
1. preflight_training_plan
- Назначение: оценка бюджета и расчет рекомендованного параллелизма.
- Может вызываться для одного генома или для набора (оценка worst-case и expected-case).

2. start_parallel_run
- Создает пустой run + scheduler.
- Возвращает run_id.

3. enqueue_genome_job
- Добавляет в run одну задачу обучения.
- Можно вызывать много раз в процессе работы run.

4. enqueue_genome_jobs
- Опциональный шорткат для добавления списка задач за один вызов.
- Не меняет семантику: внутри все равно queue-first.

5. stop_parallel_evolution
- Переводит run в stopping.
- Блокирует старт новых job.
- Отправляет cancellation активным worker.

6. get_parallel_run_status
- Возвращает queued/active/completed/cancelled, reserved budget, ошибки.

7. compute_zero_cost_score (уже есть, интегрируется в pipeline run)
- Используется как pre-queue этап.
- Возвращает proxy_score + strategy_decision.

### 8.2 События (event bus)
- training-run-started
- training-job-zero-cost-scored
- training-job-started
- training-job-progress
- training-job-finished
- training-run-stopping
- training-run-stopped
- training-run-finished
- training-run-failed

Payload везде содержит run_id + job_id + genome_id.

## 9. Изменения frontend (Evolution Studio)

### 9.1 Store/settings
Добавить настройки:
- parallelOnly: true (фиксированная константа режима)
- maxParallelJobs
- memorySafetyMarginMb
- estimatorSafetyFactor
- memorySource: auto_apu | auto_discrete | manual
- manualMemoryBudgetMb
- useZeroCostProxies
- zeroCostStrategy
- fastPassThreshold
- partialTrainingEpochs

### 9.2 UX поток
1. Пользователь нажимает Start.
2. Если включены Zero-Cost Proxies, геномы проходят proxy-оценку.
3. UI вызывает preflight_training_plan с учетом proxy decisions.
4. Если preflight reject, старт не выполняется, показать причину.
5. Если preflight OK, UI показывает:
   - estimated budget,
   - recommended parallel jobs,
   - сколько задач влезает одновременно.
6. UI вызывает start_parallel_run и enqueue_genome_job(s) только для partial/full задач.

### 9.3 Визуализация прогресса
- Убрать предположение "один текущий геном".
- Сделать таблицу/карточки active jobs.
- Метрики хранить map по job_id.
- Отдельно показывать queue depth, active count, completed count.

## 10. Корректная отмена и shutdown

### 10.1 Stop из UI
- stop_parallel_evolution(run_id)
- Scheduler прекращает admission новых задач.
- Активные worker получают cancellation token.
- После завершения всех active job run переходит в stopped.

### 10.1.1 Поведение очереди при stop
- Новые enqueue после stop отклоняются с явной ошибкой run_is_stopping.
- Задачи в pending помечаются cancelled_by_stop без запуска.
- Активные задачи завершаются как cancelled (или success, если успели финишировать до проверки токена).

### 10.2 Закрытие окна / завершение приложения
- Перехватывать close request.
- Если есть active run:
  - инициировать stop_parallel_evolution,
  - ждать bounded timeout,
  - после timeout форсировать закрытие с логом недозавершенных задач.

### 10.3 Backend cleanup checklist
- Очередь pending очищена.
- Active handles joined/aborted.
- Reserved memory = 0.
- Listener/subscription cleanup.
- Run registry удален.

## 11. Надежность и отказоустойчивость

### 11.1 Классификация ошибок
- Compile/build error генома.
- Runtime training error.
- OOM / allocation error.
- Cancellation.
- IPC/event emission error.

### 11.2 Поведение
- Ошибка одной job не валит весь run.
- Run валится только при системной ошибке оркестратора.
- Каждая job завершается terminal статусом.

### 11.3 Идемпотентность stop
- Повторный stop безопасен.
- stop на уже finished/stopped возвращает OK с текущим статусом.

## 12. Наблюдаемость

### 12.1 Логи
- run lifecycle logs
- scheduler decision logs (почему job не стартовала: budget/limit)
- reservation logs (reserve/release)
- shutdown logs

### 12.2 Метрики
- queue_wait_ms
- job_duration_ms
- active_jobs_over_time
- reserved_vram_over_time
- oom_failures_count
- cancellation_latency_ms

## 13. Пошаговый план реализации

### Этап A. Контракты и каркас
- Добавить DTO для preflight/start/stop/status/events.
- Добавить orchestrator module + RunRegistry.
- Добавить state machine run lifecycle.
- Добавить queue API: start_parallel_run + enqueue_genome_job(s).
- Добавить DTO поля для proxy_score и strategy_decision.

Acceptance:
- Компилируется, команды доступны, но scheduler пока без реального параллельного запуска.

### Этап B. Memory estimator + preflight
- Реализовать estimator по архитектуре.
- Реализовать preflight_training_plan.
- Подключить UI preflight перед стартом.
- Добавить dual preflight mode: raw vs filtered by Zero-Cost decisions.

Acceptance:
- UI может показать "не помещается" до запуска.

### Этап C. Scheduler + worker pool
- Реализовать очередь, active set, reserve/release budget.
- Реализовать dynamic admission на основе free budget.
- Подключить реальные worker jobs с параллельным выполнением.
- Реализовать событие wake-up scheduler на каждом job completion (немедленный дозапуск следующего из очереди).
- Интегрировать skip-путь: proxy-only завершение без GPU обучения.

Acceptance:
- Несколько геномов обучаются одновременно, очередь дренируется автоматически.

### Этап D. Event protocol migration
- Перевести события на run_id/job_id/genome_id.
- Переделать фронтенд-хранилище прогресса на map по job_id.
- Добавить событие proxy scoring до training-job-started.

Acceptance:
- UI корректно показывает несколько активных jobs одновременно.

### Этап E. Stop/Close cleanup hardening
- Реализовать coordinated stop.
- Реализовать close interception + graceful shutdown.
- Добавить timeout-политику.

Acceptance:
- Stop/Close не оставляют висячих задач и не держат reservation.

### Этап F. Тесты и soak
- Unit + integration + cancellation + OOM tests.
- Длительный soak (30-60 мин).

Acceptance:
- Нет утечек состояния, стабильно работает под нагрузкой.

## 14. Тест-план

### 14.1 Unit tests (backend)
- estimator для разных топологий.
- admission/release budget.
- state machine переходы run.
- stop идемпотентность.

### 14.2 Integration
- 20 jobs, budget вмещает 3 одновременно: проверить max active=3.
- Stop при active jobs: проверить terminal статусы cancelled/success.
- OOM симуляция: проверить снижение dynamic concurrency.
- Смешанный набор решений skip/partial/full: проверить, что skip не уходит в training queue.
- Проверить, что partial_train получает уменьшенный epochs.

### 14.3 Frontend
- preflight reject path.
- multi-job progress mapping.
- stop button и close behavior.

## 15. Критерии готовности (Definition of Done)

1. В коде отсутствует путь последовательного evaluate-прохода как рабочий режим.
2. Запуск эволюции всегда идет через queue + worker pool.
3. Admission based on VRAM работает и не допускает старт без резерва бюджета.
4. Stop и Close корректно чистят ресурсы и завершают run.
5. UI отображает параллельные задачи и состояние очереди.
6. Пройдены integration + soak тесты.
7. При включенном Zero-Cost корректно работают все 3 решения: skip / partial_train / full_train.

## 16. Риски и меры

Риск: estimator занижает память.
- Мера: высокий safety factor + runtime OOM feedback + auto-downscale concurrency.

Риск: фронтенд не успевает обрабатывать поток событий.
- Мера: throttling прогресса (например, событие не чаще 150-300 мс на job).

Риск: deadlock при stop/close.
- Мера: единый порядок lock acquisition + bounded waits + watchdog timeout.

## 17. Порядок внедрения в репозитории

Приоритет файлов:
1. src-tauri/src/lib.rs
2. src-tauri/src/entities.rs
3. новый модуль src-tauri/src/training/orchestrator.rs
4. src/features/evolution-studio/model/useEvolutionLoop.ts
5. src/features/evolution-manager/model/store.ts
6. src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx
7. src/widgets/title-bar/TitleBar.tsx

Ожидаемый результат после внедрения:
- Строго параллельная модель обучения с контролем VRAM, очередью, пулом воркеров и безопасным lifecycle управления ресурсами.

---

# РАЗДЕЛ II: Профилирование производительности, эволюция мультиобъектная оптимизация и расширенные возможности

Данный раздел добавляет критические функции, выявленные в анализе текущей реализации:
- Отсутству wall-clock timing и peak memory profiling
- Отсутствию механизма сохранения весов обученных моделей
- Отсутствию Pareto-фронта для мульти-objective оптимизации
- Отсутствию отслеживания генеалогии (parent-child links)
- Отсутствию скрытой библиотеки эволюционирующих геномов

## 18. Profile производительности и сбор метрик

**Текущий gap**: Training/Inference timing не отслеживается. Peak memory измеряется только оценочно.

### 18.1 Метрики для сбора

#### Per-Job Training Metrics (новые):
```rust
pub struct TrainingProfiler {
    // Timing
    pub train_start_ms: u64,
    pub first_batch_ms: u64,      // kernel compilation + first batch on GPU
    pub train_end_ms: u64,
    pub total_train_duration_ms: u64,
    
    pub val_start_ms: u64,
    pub val_end_ms: u64,
    pub val_duration_ms: u64,
    
    pub test_start_ms: u64,
    pub test_end_ms: u64,
    pub test_duration_ms: u64,
    
    // Memory (дополнительно к estimate)
    pub peak_active_memory_mb: f32,    // max concurrent allocations during training
    pub peak_model_params_mb: f32,     // frozen after first forward pass
    pub peak_gradient_mb: f32,         // peak during backward
    pub peak_optim_state_mb: f32,      // peak Adam state
    pub peak_activation_mb: f32,       // peak intermediate tensors
    
    // Throughput
    pub samples_per_sec: f32,          // training batches
    pub inference_msec_per_sample: f32, // val/test average latency
    
    // Quality-of-life
    pub batch_count: u32,
    pub early_stop_epoch: Option<u32>, // если был early stopping
}
```

#### Aggregate Generation-Level Metrics:
```rust
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

### 18.2 Instrumentation points (backend)

#### Точка 1: Training Entry (entities.rs, run_eval_pass)
```rust
let train_start = Instant::now();
let first_batch_instant = first_batch_indicator;
// ... training loop
// profile.train_end_ms = train_start.elapsed().as_millis();
```

#### Точка 2: Memory tracking (entities.rs per layer)
```rust
// After each layer output computation:
let current_memory = measure_allocator_usage(); // bind to allocator profiler
profile.peak_active_memory_mb = max(profile.peak_active_memory_mb, current_memory);
```

Burn integration если доступно:
- Burn Backend может exposeMemoryStats через GraphBuilder inspector.
- Fallback: запросить WGPU Device statistics (если поддерживает).

#### Точка 3: Inference timing (run_validation_pass)
```rust
let batch_time_start = Instant::now();
let _ = model.forward(batch);
let batch_time_ms = batch_time_start.elapsed().as_millis();
total_inference_time_ms += batch_time_ms;
```

#### Точка 4: Per-job result instrumentation
```rust
TrainingResult {
    job_id,
    genome_id,
    status: TrainingStatus::Success,
    loss,
    accuracy,
    profiler,  // NEW: TrainingProfiler with all metrics
    // ... existing fields
}
```

### 18.3 UI Integration (frontend)

#### New generation-level table columns:
- Training Time Total
- Avg Inference Time (ms)
- Peak VRAM (Mb)
- Samples/sec throughput

#### New genome detail card (modal):
- Shows TrainingProfiler breakdown (train/val/test durations separately)
- Memory profile chart (peak active vs estimate)
- Early-stop info if applicable

#### Genome comparison view:
- Select 2-3 genomes, compare side-by-side:
  - Accuracy vs Training Time scatter
  - Accuracy vs Peak Memory scatter

### 18.4 Events emission
```
training-job-profiler-available
  - payload: { job_id, genome_id, profiler: TrainingProfiler }
  - timing: when job completes (before training-job-finished)
```

### 18.5 Storage (hidden library)
- Profiler data preserved alongside genome JSON in library entry
- Enables later analysis of performance trends

---

## 19. Multi-Objective Optimization: Pareto Front

**Текущий gap**: Single scalarized fitness. Нет Pareto dominance relation и frontier computation.

### 19.1 Objective space definition

Парето-фронт по 3D пространству:
- **F1: Accuracy** (maximize)
- **F2: Latency** (minimize) - Inference ms per sample
- **F3: Model Size** (minimize) - MB (encoded as param count)

Опционально:
- F4: Training Speed (multiply samples/sec, max)
- F5: Energy (if device profile includes power budget)

Рекомендуемый вариант: 3D (Accuracy × Latency × Size).

### 19.2 Dominance relation

Genome A dominates B if:
- A.accuracy >= B.accuracy AND
- A.inference_latency <= B.inference_latency AND  
- A.model_size <= B.model_size AND
- at least one inequality is strict

```rust
pub fn is_dominated(a: &GenomeMetrics, b: &GenomeMetrics) -> bool {
    // true if a is dominated by b
    (b.accuracy >= a.accuracy) &&
    (b.inference_latency <= a.inference_latency) &&
    (b.model_size <= a.model_size) &&
    !(a.accuracy == b.accuracy && 
      a.inference_latency == b.inference_latency && 
      a.model_size == b.model_size)
}
```

### 19.3 Pareto front computation

После каждого поколения:
```rust
pub fn compute_pareto_front(generation: &[GenomeMetrics]) -> Vec<GenomeMetrics> {
    let mut front = Vec::new();
    for candidate in generation {
        if !front.iter().any(|frontier_member| is_dominated(candidate, frontier_member)) {
            // Remove dominated members
            front.retain(|m| !is_dominated(m, candidate));
            front.push(candidate.clone());
        }
    }
    front
}
```

Complexity: O(N^2) per generation, где N = population size. For N < 200: acceptable.

### 19.4 Backend API extension

#### New DTO: GenomeObjectives
```rust
pub struct GenomeObjectives {
    pub genome_id: String,
    pub accuracy: f32,
    pub inference_latency_ms: f32,  // from profiler
    pub model_size_mb: f32,          // computed from network structure
    pub training_time_ms: u64,       // from profiler
    pub is_dominated: bool,
    pub domination_count: u32,       // how many genomes dominate this one
}
```

#### New response: GenerationParetoFront
```rust
pub struct GenerationParetoFront {
    pub generation: u32,
    pub total_genomes: u32,
    pub pareto_members: Vec<GenomeObjectives>,  // non-dominated set
    pub objectives_3d: Vec<(f32, f32, f32)>,   // (accuracy, latency, size)
}
```

#### New command:
```
compute_pareto_front
  - input: run_id, generation_number
  - output: GenerationParetoFront
  - timing: called post-evaluation of generation
```

### 19.5 Frontend: Pareto Visualization

#### 2D Scatter Plot (Accuracy vs Latency)
- X-axis: Inference Latency (ms)
- Y-axis: Accuracy (%)
- Size: proportional to model size
- Highlight: non-dominated genomes (bold, unique color)
- Tooltip: accuracy, latency, size, parent_id, generation

#### 3D Scatter Plot (optional, if 3D canvas library available)
- X: Latency, Y: Accuracy, Z: Model Size
- Interactive rotation/zoom
- Color gradient: fitness value

#### Evolution Phase Display
- "Generation 5: 15 genomes, 4 on Pareto front"
- Timeline showing front growth over generations

#### Selection from Front
- Click on non-dominated genome
- Option: "Use as next seed" or "Export & Save weights"

### 19.6 Store updates (frontend)

```typescript
interface GenerationSnapshot {
  // ... existing fields
  paretoFront?: GenomeObjectives[];
  objectiveSpace?: {
    accuracy: { min: number; max: number };
    latency: { min: number; max: number };
    modelSize: { min: number; max: number };
  };
}

interface EvolutionState {
  // ... existing
  paretoHistory: Map<generationNumber, GenerationParetoFront>;
  currentParetoFront: GenomeObjectives[];
}
```

---

## 20. Device Profile System

**Goal**: Permit evolution to target specific devices (embedded, edge, cloud) with resource constraints.

### 20.1 Device Profile Schema

```rust
pub struct DeviceProfile {
    pub device_id: String,  // "embedded-arm32", "edge-gpu-jetson", "laptop-cpu", etc.
    pub device_name: String,
    pub compute_capability: ComputeType,        // ARM, x86, GPU
    pub ram_mb: u32,
    pub vram_mb: Option<u32>,  // GPU memory if available
    pub inference_latency_budget_ms: f32,       // constraint for target device
    pub training_available: bool,
    pub power_budget_mw: Option<u32>,           // milliwatts if available
    pub max_model_size_mb: Option<f32>,         // model compression constraint
    pub target_fps: Option<f32>,                // for real-time apps
}
```

### 20.2 Built-in Device Profiles

Predefined:
1. **embedded-microcontroller**: 256MB RAM, no GPU, 50ms latency budget, 1MB model size
2. **embedded-arm64**: 1GB RAM, no GPU, 200ms latency budget, 50MB model size
3. **mobile-phone**: 4GB RAM, optional GPU, 100ms latency, 100MB model
4. **edge-gateway**: 8GB RAM, 2GB VRAM, 500ms latency, 500MB model
5. **laptop-cpu**: 16GB RAM, 200ms latency, 2GB model
6. **laptop-gpu**: 16GB RAM, 8GB VRAM, 50ms latency, 2GB model
7. **desktop-cpu**: unlimited
8. **desktop-gpu**: unlimited
9. **cloud-gpu**: unlimited

### 20.3 Device-Aware Fitness

При выборе device profile, fitness computation:
```rust
adjusted_fitness = base_fitness
    * device_constraint_penalty(inference_latency_ms, profile.latency_budget)
    * device_constraint_penalty(model_size_mb, profile.max_model_size)
    - parsimony_alpha * node_count
```

Где constraint_penalty:
```rust
fn device_constraint_penalty(actual: f32, budget: f32) -> f32 {
    if actual <= budget {
        1.0  // within budget, no penalty
    } else {
        let excess_ratio = actual / budget;
        (1.0 / excess_ratio).powf(2.0)  // quadratic penalty
    }
}
```

### 20.4 Frontend: Device Profile Selector

#### Pre-evolution panel:
- Dropdown: "Target Device"
- If custom: input fields for RAM, VRAM, latency_budget, max_model_size
- Info: "Recommended parallelism for {device}: N genomes"

#### During evolution:
- Display: "Optimizing for {device_name}, latency budget {X ms}"
- Show genomes violating constraints separately

#### Post-evolution Pareto:
- Highlight genomes meeting all device constraints (feasible)
- Gray out infeasible genomes
- Show "best feasible" candidate

### 20.5 Store integration

```typescript
interface EvolutionSettings {
  // ... existing
  deviceProfile: DeviceProfile;
  isCustomDevice: boolean;
  customDeviceParams?: { ram_mb: number; vram_mb?: number; latency_budget_ms: number; max_model_size_mb?: number };
}
```

---

## 21. Genealogy Tracking: Family Tree

**Current gap**: No parent-child links, no mutation history, no ancestral lineage.

### 21.1 Genealogy Data Structures

```rust
pub struct GenomeGeneology {
    pub genome_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,  // 1 or 2 (mutation vs crossover)
    pub mutation_type: MutationType,  // AddNode, RemoveNode, RemoveSubgraph, ParameterMutation, Crossover, etc.
    pub mutation_params: serde_json::Value,  // details of mutation applied
    pub fitness: f32,
    pub accuracy: f32,
    pub created_at_ms: u64,
}

pub enum MutationType {
    Random,       // initial random population
    AddNode { node_type: String, source: String, target: String },
    RemoveNode { node_id: String },
    RemoveSubgraph { node_ids: Vec<String> },
    ParameterMutation { layer_id: String, param_name: String, old_value: serde_json::Value, new_value: serde_json::Value },
    ParameterScale { layer_id: String, scale_factor: f32 },
    Crossover { parent1: String, parent2: String, contribution: String },
}
```

### 21.2 Generation Entry Extended

```rust
pub struct GenerationSnapshot {
    // ... existing
    pub genealogy: HashMap<String, GenomeGeneology>,  // genome_id -> genealogy
}
```

Frontend:
```typescript
interface PopulatedGenome {
  // ... existing fields
  generation: number;
  parent_ids?: string[];
  mutation_type?: string;
  mutation_params?: Record<string, any>;
}
```

### 21.3 Mutation Tracking Integration

In useEvolutionLoop.ts:
```typescript
const runGeneration = async () => {
  // ... existing logic

  // After breeding step:
  offspringGenome.generation = currentGeneration;
  offspringGenome.parent_ids = [parent1.id, parent2.id];
  offspringGenome.mutation_type = lastMutationApplied;
  offspringGenome.mutation_params = mutationDetails;

  // Store in genealogy map
  generationHistory[i].genealogy[offspringGenome.id] = {
    genome_id: offspringGenome.id,
    generation: currentGeneration,
    parent_ids: offspringGenome.parent_ids,
    // ...
  };
};
```

### 21.4 Genealogy Tree Viewer (UI)

#### New component: GenealogiesTreeView
- Tree layout: root = generation 0 randoms, edges = ancestry links
- Hover on node: show mutation applied, fitness change
- Click: open genome details + compare with parents
- Filter: show only genomes on Pareto front, or filtered by fitness range
- Export: GraphML or JSON format for external analysis

#### Mini tree in evolution stats:
- Timeline of top-N genomes per generation
- Highlight:paths from initial population to current best
- Show mutation types as edge labels

---

## 22. Hidden Genome Library & Weight Persistence

**Current gap**: No automatic archival of evolved genomes, no weight checkpointing, no hidden-flag mechanism.

### 22.1 Hidden Library Storage

Extend current genome library with:

```rust
pub struct RustGenomeLibraryEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub genome_json: String,
    pub created_at: String,
    pub is_pinned: bool,
    pub is_hidden: bool,            // NEW
    pub source_generation: Option<u32>, // NEW: which evolution run/generation
    pub parent_genomes: Vec<String>,  // NEW: for genealogy traceability
    pub fitness_metrics: Option<GenomeObjectives>,  // NEW: accuracy, latency, size from evolution
    pub profiler_data: Option<TrainingProfiler>,    // NEW: timing/memory from evolution
    pub model_weights: Option<String>,              // NEW: path to .safetensors or weight file
    pub device_profile_target: Option<DeviceProfile>, // NEW: which device optimized for
}
```

### 22.2 Auto-save During Evolution

During each generation evaluation:
```rust
for genome in completed_genomes {
    let entry = RustGenomeLibraryEntry {
        id: format!("{}-gen{}-{}", run_id, generation, genome_id),
        name: format!("Gen {} - {}", generation, genome_id),
        genome_json: serialize_genome(&genome),
        is_hidden: true,  // start as hidden
        source_generation: Some(generation),
        parent_genomes: genome.genealogy.parent_ids.clone(),
        fitness_metrics: Some(genome.objectives.clone()),
        profiler_data: Some(genome.profiler.clone()),
        model_weights: None,  // initially, weights not saved
        // ...
    };
    save_to_library(entry);
}
```

### 22.3 Weight Checkpointing (Post-Evolution Selection)

After Pareto front shown, user selects genome and clicks "Export with Weights":

```rust
pub async fn export_genome_with_weights(genome_id: String, output_path: String) -> Result<()> {
    let entry = library.get(genome_id)?;
    let genome = deserialize_genome(&entry.genome_json)?;
    
    // Re-train or load from cached weights
    let (model, loss, accuracy) = rebuild_and_train_genome(&genome, full_epochs)?;
    
    // Save weights using Burn's .save() API
    let weights_path = format!("{}/genome-weights.safetensors", output_path);
    model.save(&weights_path)?;
    
    // Save metadata
    let metadata = serde_json::json!({
        "genome_id": genome_id,
        "accuracy": accuracy,
        "loss": loss,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "device_profile": entry.device_profile_target,
    });
    std::fs::write(format!("{}/metadata.json", output_path), serde_json::to_string_pretty(&metadata)?)?;
    
    Ok(())
}
```

### 22.4 Hidden Library UI

#### New page: "Hidden Archive" (accessible from main menu)
- Searchable table of all auto-saved genomes
- Columns: Generation, Fitness, Accuracy, Latency, Parent Genomes, created_at
- Filter: generation range, fitness min/max, accuracy range, device profile
- Bulk ops: select multiple, unhide all, delete all, export batch
- Individual: clone as new seed, unhide & use in next evolution, export with weights

#### Unhide workflow:
1. User right-clicks genome or uses "Unhide" button
2. Genome moves to public library, gets default name assignment
3. Can use as initial seed in next evolution run

#### Export workflow:
1. Select genome from hidden or public library
2. Click "Export Model Weights"
3. Dialog: select output folder
4. Backend re-trains (or loads cached weights if available) + saves .safetensors + metadata.json
5. Options: include genome JSON, include history/genealogy

### 22.5 Store/API Changes

#### Frontend store:
```typescript
interface LibraryState {
  publicGenomes: PopulatedGenome[];
  hiddenGenomes: PopulatedGenome[];
  loadVisibleGenomes: () => void;
  loadHiddenGenomes: () => void;
  unhideGenome: (id: string) => void;
  deleteGenome: (id: string) => void;
  exportWithWeights: (id: string, outputPath: string) => Promise<void>;
}
```

#### Backend commands:
- `list_hidden_library_genomes`
- `unhide_library_genome`
- `delete_hidden_library_entry`
- `export_genome_with_weights(genome_id, output_path)`

---

## 23. User-Defined Stopping Criteria

**Current gap**: Only generation count limit. No plateau detection, no time budget, no manual override with easy undo.

### 23.1 Stopping Criteria Types

```rust
pub enum StoppingCriterion {
    GenerationLimit { max_generations: u32 },
    
    FitnessPlateau {
        patience_generations: u32,  // iters with no improvement
        improvement_threshold: f32, // min improvement to reset patience
        monitor: "best_fitness" | "pareto_coverage" | "population_avg",
    },
    
    TimeLimit { max_seconds: u32 },
    
    TargetAccuracy { threshold: f32 }, // if accuracy >= threshold, stop and export
    
    ManualStop,  // user initiated
}

pub struct StoppingPolicy {
    pub criteria: Vec<StoppingCriterion>,
    pub policy_type: "any" | "all",  // stop on ANY or ALL criteria met
}
```

### 23.2 Plateau Detection

```rust
pub fn check_fitness_plateau(
    history: &[GenerationSnapshot],
    criterion: &FitnessPlateau,
) -> (bool, u32) {
    let recent = history.last(criterion.patience_generations as usize).unwrap_or(&[]);
    let best_in_recent = recent.iter().map(|g| g.best_fitness).max().unwrap_or(0.0);
    let prev_best = history.get(history.len() - criterion.patience_generations as usize - 1)
        .map(|g| g.best_fitness)
        .unwrap_or(0.0);
    
    let improvement = (best_in_recent - prev_best) / (prev_best + 1e-6);
    let is_plateau = improvement < criterion.improvement_threshold;
    
    (is_plateau, criterion.patience_generations)
}
```

### 23.3 Frontend: Stopping Criteria Panel

#### Pre-evolution configuration:
- Checkboxes for each criterion type
- Inputs for parameters (max_generations, patience, time_limit_sec, target_accuracy)
- Dropdown: "Stop on" = [Any criterion met / All criteria met]
- Visual: chart showing estimated time/generations if possible

#### During evolution:
- Live progress bar: generations completed vs max_gen limit
- Live timer: elapsed time vs time limit
- Fitness plot: show current trend, plateau detection visualization
- Badge: which criteria closest to threshold

#### On stop:
- Dialog: which criterion triggered stop
- Options: continue, export & save, view Pareto front

### 23.4 Backend: Criterion Evaluation

Per-generation:
```rust
pub fn check_all_criteria(
    run: &TrainingRun,
    generation: u32,
    history: &[GenerationSnapshot],
    policy: &StoppingPolicy,
) -> (bool, Vec<String>) {
    let mut triggered_criteria = Vec::new();
    
    for criterion in &policy.criteria {
        let should_stop = match criterion {
            StoppingCriterion::GenerationLimit { max_generations } => {
                generation >= *max_generations
            }
            StoppingCriterion::FitnessPlateau { ... } => {
                check_fitness_plateau(history, criterion).0
            }
            StoppingCriterion::TimeLimit { max_seconds } => {
                run.elapsed_seconds() >= *max_seconds
            }
            StoppingCriterion::TargetAccuracy { threshold } => {
                history.last().map(|g| g.best_fitness >= *threshold).unwrap_or(false)
            }
            StoppingCriterion::ManualStop => false,  // handled separately
        };
        
        if should_stop {
            triggered_criteria.push(format!("{:?}", criterion));
        }
    }
    
    let stop_now = match policy.policy_type {
        "any" => !triggered_criteria.is_empty(),
        "all" => triggered_criteria.len() == policy.criteria.len(),
    };
    
    (stop_now, triggered_criteria)
}
```

### 23.5 Store integration

```typescript
interface EvolutionSettings {
  // ... existing
  stoppingCriteria: StoppingCriterion[];
  stoppingPolicy: "any" | "all";
}

interface EvolutionState {
  // ... existing
  currentStoppingProgress: {
    generationsSoFar: number;
    elapsedSeconds: number;
    plateauPatience: number;
    bestAccuracySoFar: number;
    triggeredCriteria?: string[];
  };
}
```

---

## 24. User Interface Comprehensive Specifications

**New pages and panels for all features.**

### 24.1 Pre-Evolution Setup Panel (expanded)

#### Section: Basic Settings
- Dataset selector (already exists)
- Population size: 10-500 (slider + input)
- Max generations: 1-1000 (or unlimited + time limit)

#### Section: Performance & Objectives (NEW)
- Mode selector: [Single-Objective (legacy) | Multi-Objective (Pareto)]
- If Pareto:
  - "Primary Objective": Accuracy (always)
  - "Secondary Objectives": checkboxes for [Latency, Model Size, Training Time, Energy]
  - Weights/emphasis sliders (optional)

#### Section: Device Targeting (NEW)
- Dropdown: "Target Device"
- Pre-canned options: [Embedded, Mobile, Laptop, Desktop, Cloud, Custom]
- If Custom: fields for RAM, VRAM, latency_budget, max_model_size

#### Section: Stopping Criteria (NEW)
- Add criterion button
- List of active criteria with removable X
- Per criterion:
  - Type dropdown [GenerationLimit, FitnessPlateau, TimeLimit, TargetAccuracy]
  - Parameter input fields

#### Section: Zero-Cost Options (existing, refactor)
- Checkbox: "Use Zero-Cost Proxies"
- If checked:
  - Strategy dropdown [skip / partial_train / full_train]
  - Thresholds inputs

#### Section: Parallelism (NEW, from orchestrator)
- "Max Parallel Jobs": 1-16 (slider)
- After dataset + genome count input:
  - Button "Check Memory Budget"
  - Shows: estimated VRAM per genome, recommended parallelism, warning if tight

#### Action Buttons:
- "Start Evolution"
- "Load Previous Run" (optional)

### 24.2 Evolution Progress Dashboard (live, during run)

#### Top Section: Overview Cards
- Generations Elapsed: X / Y
- Genomes Evaluated: X / (population_size × generations)
- Current Best Fitness: 0.XX
- Pareto Front Size: N genomes (if MOO enabled)
- Elapsed Time: HH:MM:SS
- ETA: (if plateau detection available)

#### Middle Section: Real-Time Metrics Chart
- Line plot: Best Fitness over generations (primary Y)
- Area plot: Population average fitness (secondary Y)
- Shaded region: valid device constraints (if device profile set)
- Hover: tooltip with exact values

#### Right Side Panel: Active Jobs
- Table (if parallelism > 1):
  - Job ID, Genome ID, Progress (%), Status, Duration, ETA
  - Color code: running (blue), completed (green), failed (red), queued (gray)
- If parallelism == 1: simplified view (just current job)

#### Bottom Section (tabs):
- **Tab 1: Pareto Front (if MOO)**: 2D scatter plot
  - X: Latency, Y: Accuracy, Size: model_size
  - Highlighted: non-dominated set
  - Tooltip: accuracy, latency, size, generation

- **Tab 2: Genealogy Tree (if genealogy enabled)**: interactive tree
  - Nodes: genomes by generation
  - Edges: parent-child relationships labeled with mutation type
  - Color: fitness value gradient
  - Click node: show genome details

- **Tab 3: Stopping Criteria Progress**: 
  - For each active criterion: progress bar
  - Plateau: patience meter (X / patience_value generations)
  - Time limit: timer
  - Target accuracy: progress bar toward threshold

- **Tab 4: Performance Metrics** (NEW)
  - Per genome: Training time, Inference Latency, Peak VRAM
  - Charts: training time vs accuracy scatter, memory vs accuracy

#### Control Buttons:
- "Pause Evolution" (pauses job scheduling, preserves state)
- "Resume Evolution"
- "Stop Evolution" (triggers stopping sequence)
- "Save Checkpoint" (exports current hidden library as zip)

### 24.3 Post-Evolution Analysis Panel (after run completes)

#### Section: Pareto Front Visualization (MOO only)
- 2D/3D scatter plot (same as during-evolution)
- Legend: feasible (meets all device constraints), infeasible
- Selection: click on genome → details pane

#### Section: Detailed Genome Comparison (NEW)
- Multi-select from Pareto front (2-3 genomes)
- Side-by-side comparison:
  - Architecture (layer counts, types)
  - Accuracy / Latency / Model Size
  - Training time / Inference latency
  - Peak memory
  - Genealogy (parent chain)
  - Device profile compatibility badge

#### Section: Genealogy Analysis (NEW)
- Interactive family tree visualization
- Timeline: generation → fitness progression
- Hover on genome: show mutations leading to it
- Click: export ancestral lineage as JSON/GraphML

#### Section: Hidden Library Management (NEW)
- "N genomes auto-saved to hidden archive"
- Button: "View Archive"
- Quick stats: average fitness, accuracy range
- Bulk actions: "Unhide all", "Delete all", "Export batch"

#### Export/Action Buttons:
- "Download Pareto Front (JSON)"
- "Select & Export Model Weights" → opens weight export dialog
- "Save Evolution Report (PDF)" → summary report
- "Continue Evolution" → re-use Pareto front as new population seed

### 24.4 Hidden Archive Page (NEW)

#### Header:
- "Hidden Genome Archive: N genomes from M evolution runs"

#### Table: searchable, sortable
- Genome ID
- Generation (run #, gen within run)
- Accuracy
- Latency (ms)
- Model Size (MB)
- Parent IDs
- Created (date/time)
- Device Profile Target
- Actions: [Unhide, Clone as Seed, Export Weights, Delete, View Details]

#### Filters (sidebar):
- Generation range slider
- Accuracy min/max
- Latency range
- Model size range
- Device profile dropdown
- Search text box (genome ID, parent ID, generation)

#### Bulk Actions (top):
- Select all / select none
- "Unhide Selected"
- "Delete Selected"
- "Export Batch as ZIP"

#### Detail Modal (click genome):
- Full genealogy chain (parents, grandparents, ..., generation 0)
- Accuracy / Latency / Size metrics
- Profiler breakdown (training time, val time, peak memory)
- Network architecture diagram
- Fitness trajectory (if from Pareto tracking)

### 24.5 Settings Panel Updates (Evolution Manager)

#### Tab: "Advanced Performance"
- Memory safety margin: 256-2048 MB (slider)
- Estimator safety factor: 1.0-2.0 (slider)
- Memory source: [Auto (APU/Discrete), Manual]
- Manual memory budget: (if Manual selected)
- Profiling enabled: checkbox → collect timing + memory data

#### Tab: "Multi-Objective" (NEW)
- Objective space: 3D [Accuracy, Latency, Model Size]
- Weights/emphasis: sliders for each objective (if preferences available)
- Reference device: selector → show which device profile used for constraints

#### Tab: "Device Profiles" (NEW)
- Built-in profiles: dropdown list [Embedded, Mobile, Laptop, Desktop, Cloud]
- Custom profiles: manage button → add/edit/delete custom profiles
- Active profile during evolution: radio button selector

---

## 25. Test Plan: Unit, Integration, E2E

### 25.1 Backend Unit Tests (Rust)

#### Module: performance_profiler

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_profiler_timing_collection() {
        // Create dummy TrainingProfiler, check timing fields populated
        let profiler = TrainingProfiler::new();
        profiler.mark_train_start();
        std::thread::sleep(Duration::from_millis(100));
        profiler.mark_train_end();
        assert!(profiler.total_train_duration_ms >= 100);
    }
    
    #[test]
    fn test_memory_tracking() {
        // Simulate peak memory updates, verify max is tracked
    }
    
    #[test]
    fn test_throughput_calculation() {
        // samples_per_sec = batch_count * batch_size / train_duration_ms
    }
}
```

#### Module: pareto_front

```rust
#[test]
fn test_dominance_relation() {
    let a = GenomeMetrics { accuracy: 0.95, latency_ms: 50.0, size_mb: 10.0 };
    let b = GenomeMetrics { accuracy: 0.90, latency_ms: 100.0, size_mb: 20.0 };
    assert!(is_dominated(&b, &a));  // a dominates b
    assert!(!is_dominated(&a, &b)); // a not dominated by b
}

#[test]
fn test_pareto_front_computation() {
    let genomes = vec![
        GenomeMetrics { accuracy: 0.95, latency_ms: 50.0, size_mb: 10.0 },
        GenomeMetrics { accuracy: 0.90, latency_ms: 40.0, size_mb: 15.0 },
        GenomeMetrics { accuracy: 0.92, latency_ms: 60.0, size_mb: 12.0 },
    ];
    let front = compute_pareto_front(&genomes);
    assert_eq!(front.len(), 2); // Two non-dominated
}

#[test]
fn test_pareto_empty_and_single() {
    assert_eq!(compute_pareto_front(&vec![]), 0);
    assert_eq!(compute_pareto_front(&vec![...single...]), 1);
}
```

#### Module: device_profile

```rust
#[test]
fn test_device_constraint_penalty() {
    // if actual <= budget: penalty = 1.0
    assert_eq!(device_constraint_penalty(100.0, 200.0), 1.0);
    // if actual > budget: penalty = (1.0 / excess_ratio)^2
    assert_eq!(device_constraint_penalty(200.0, 100.0), 0.25);
}

#[test]
fn test_built_in_profiles_validation() {
    for profile in BUILT_IN_PROFILES {
        assert!(profile.ram_mb > 0);
        assert!(profile.inference_latency_budget_ms > 0.0);
    }
}
```

#### Module: genealogy_tracker

```rust
#[test]
fn test_genealogy_chain_validity() {
    // Create genome lineage, verify parent_ids resolve
    // Verify no cycles
}

#[test]
fn test_mutation_type_serialization() {
    let mutations = vec![
        MutationType::AddNode { ... },
        MutationType::ParameterMutation { ... },
        MutationType::Crossover { ... },
    ];
    for m in mutations {
        let serialized = serde_json::to_string(&m).unwrap();
        let deserialized: MutationType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(m, deserialized);
    }
}
```

#### Module: weight_checkpoint

```rust
#[test]
fn test_weights_export_to_safetensors() {
    // Build dummy model, export weights, verify file created
    // Verify metadata.json includes accuracy/loss
}

#[test]
fn test_hidden_library_entry_storage() {
    // Create entry, save, load, verify fields
}
```

#### Module: stopping_criteria

```rust
#[test]
fn test_fitness_plateau_detection() {
    let history = vec![...genomes with plateau...];
    let criterion = FitnessPlateau { patience_generations: 5, improvement_threshold: 0.01, ... };
    let (is_plateau, _) = check_fitness_plateau(&history, &criterion);
    assert!(is_plateau);
}

#[test]
fn test_time_limit_criterion() {
    // Simulate elapsed time, verify criterion triggered
}

#[test]
fn test_stopping_policy_any_vs_all() {
    // Test "any" policy: stop if ANY criterion triggered
    // Test "all" policy: stop if ALL criteria triggered
}
```

### 25.2 Frontend Unit Tests (Vitest + jsdom)

#### Module: pareto front computation (TypeScript)

```typescript
import { computeParetoFront, isDominated } from '@/shared/lib/pareto';

describe('Pareto Front', () => {
  it('should identify dominated genomes', () => {
    const a = { accuracy: 0.95, latency_ms: 50, size_mb: 10 };
    const b = { accuracy: 0.90, latency_ms: 100, size_mb: 20 };
    expect(isDominated(b, a)).toBe(true);
  });

  it('should compute Pareto front for sample population', () => {
    const genomes = [
      { accuracy: 0.95, latency_ms: 50, size_mb: 10 },
      { accuracy: 0.90, latency_ms: 40, size_mb: 15 },
      { accuracy: 0.92, latency_ms: 60, size_mb: 12 },
    ];
    const front = computeParetoFront(genomes);
    expect(front.length).toBe(2);
  });
});
```

#### Module: device constraint penalty

```typescript
import { deviceConstraintPenalty } from '@/shared/lib/device-profiles';

describe('Device Constraints', () => {
  it('should apply no penalty within budget', () => {
    expect(deviceConstraintPenalty(100, 200)).toBe(1.0);
  });

  it('should apply quadratic penalty over budget', () => {
    expect(deviceConstraintPenalty(200, 100)).toBe(0.25);
  });
});
```

#### Module: stopping criteria frontend

```typescript
import { checkStoppingCriteria } from '@/features/evolution-manager/model/stopping-criteria';

describe('Stopping Criteria', () => {
  it('should detect generation limit', () => {
    const criteria = [{ type: 'GenerationLimit', max_generations: 10 }];
    expect(checkStoppingCriteria(criteria, 10, [])).toBe(true);
    expect(checkStoppingCriteria(criteria, 9, [])).toBe(false);
  });

  it('should detect fitness plateau', () => {
    const history = [0.5, 0.6, 0.65, 0.66, 0.665, 0.666]; // plateau after 3
    const criterion = { type: 'FitnessPlateau', patience: 3, threshold: 0.01 };
    expect(checkStoppingCriteria([criterion], 6, history)).toBe(true);
  });
});
```

#### Module: genealogy tree rendering

```typescript
import { renderGenealogicTree } from '@/features/genealogy-viewer/ui';

describe('Genealogy Tree', () => {
  it('should build parent-child edges correctly', () => {
    const genomes = [
      { id: 'g1', parents: [] },
      { id: 'g2', parents: ['g1'] },
      { id: 'g3', parents: ['g1', 'g2'] }, // crossover
    ];
    const edges = buildEdges(genomes);
    expect(edges.length).toBe(2);
  });

  it('should detect cycles (impossible in valid genealogy)', () => {
    // circular parent refs should be caught
  });
});
```

### 25.3 Integration Tests

#### Scenario 1: Multi-Objective Evolution End-to-End
```rust
#[tokio::test]
async fn test_moo_evolution_complete_cycle() {
    // 1. Start evolution run with MOO enabled
    // 2. Enqueue 10 genomes
    // 3. Run 5 generations
    // 4. Verify Pareto front computed after each generation
    // 5. Verify objectives (accuracy, latency, size) tracked
    // 6. Verify profiler data collected
    // 7. Stop and verify shutdown clean
}
```

#### Scenario 2: Device-Aware Fitness
```rust
#[tokio::test]
async fn test_device_profile_fitness_filtering() {
    // 1. Set device profile to "embedded-microcontroller"
    // 2. Evolve population
    // 3. Verify genomes violating latency/size constraints get penalized
    // 4. Verify Pareto front only contains feasible genomes
}
```

#### Scenario 3: Weight Checkpointing + Hidden Library
```rust
#[tokio::test]
async fn test_auto_save_and_weight_export() {
    // 1. Run evolution 2 generations
    // 2. Verify all genomes auto-saved to hidden library
    // 3. Export best genome with weights
    // 4. Verify .safetensors + metadata.json files created
    // 5. Load weights, verify model inference works
}
```

#### Scenario 4: Genealogy Tracking
```rust
#[tokio::test]
async fn test_genealogy_across_generations() {
    // 1. Start with random population (0 parents)
    // 2. Generation 1: mutations + crossover
    // 3. Verify parent_ids set correctly
    // 4. Verify mutation types logged
    // 5. Build genealogy tree, verify no cycles
}
```

#### Scenario 5: Stopping Criteria Integration
```rust
#[tokio::test]
async fn test_stopping_criteria_triggered() {
    // 1. Set stopping criteria: ["FitnessPlateau" with patience=3, "TimeLimit" 30s]
    // 2. Run evolution
    // 3. If plateau detected first: verify stop, confirm criterion
    // 4. If time limit first: verify stop, confirm criterion
}
```

#### Scenario 6: Parallelism + Profiling
```rust
#[tokio::test]
async fn test_parallel_jobs_profiling() {
    // 1. Set max_parallel_jobs = 3
    // 2. Enqueue 10 genomes
    // 3. Verify max 3 active simultaneously
    // 4. Verify timing + memory profiling on all jobs
    // 5. Verify VRAM reservation/release cycle
}
```

### 25.4 E2E Tests (Tauri + Frontend)

#### Test 1: Complete Evolution Workflow with MOO
```typescript
describe('Complete MOO Evolution E2E', () => {
  it('should run full evolution with Pareto front', async () => {
    // 1. Load dataset
    // 2. Enter settings: pop=20, gens=3, target_device=mobile, MOO=true
    // 3. Click Start
    // 4. Monitor progress panels (jobs, fitness chart, genealogy tree)
    // 5. Verify Pareto front updates per generation
    // 6. Stop evolution
    // 7. View post-evolution analysis
    // 8. Select genome, export weights
    // 9. Verify .safetensors + metadata.json downloaded
  });
});
```

#### Test 2: Stopping Criteria Pause/Resume
```typescript
describe('Stopping Criteria E2E', () => {
  it('should pause/resume and stop on plateau', async () => {
    // 1. Configure: FitnessPlateau (patience=2)
    // 2. Start evolution
    // 3. Pause after gen 2
    // 4. Resume, run gen 3-4
    // 5. If plateau: verify stop triggered with reason message
  });
});
```

#### Test 3: Hidden Archive Unhide & Reseed
```typescript
describe('Hidden Archive E2E', () => {
  it('should unhide genome and use as seed', async () => {
    // 1. Run first evolution (20 gens), stop
    // 2. Check archive: N genomes auto-saved, all hidden
    // 3. Open archive page, search for best genome
    // 4. Click "Unhide & Use as Seed"
    // 5. Start new evolution (seed population includes unhidden genome)
    // 6. Verify genealogy links both runs
  });
});
```

#### Test 4: Device Profile Constraint Visualization
```typescript
describe('Device Profile Constraints E2E', () => {
  it('should highlight feasible vs infeasible genomes', async () => {
    // 1. Select embedded device (8MB max, 50ms latency budget)
    // 2. Run evolution
    // 3. In Pareto front view, verify:
    //    - Feasible genomes highlighted (green)
    //    - Infeasible grayed out (red)
    // 4. Filter to show only feasible
    // 5. Export shows only feasible on front
  });
});
```

#### Test 5: Weight Export Workflow
```typescript
describe('Weight Export E2E', () => {
  it('should export model weights with metadata', async () => {
    // 1. Run evolution
    // 2. View Pareto front
    // 3. Select top genome
    // 4. Click "Export Model Weights"
    // 5. Dialog: choose folder, confirm
    // 6. Verify zip contains:
    //    - genome.json
    //    - model-weights.safetensors
    //    - metadata.json (accuracy, loss, timestamp, profiler data)
    // 7. Load .safetensors in external tool, verify valid
  });
});
```

### 25.5 Stress & Soak Tests

#### Soak Test 1: Long-running evolution
```
- Population: 50 genomes
- Generations: 100+
- Duration: 1-2 hours
- Targets: memory stability, no leak, no performance degradation
```

#### Soak Test 2: Archive size limits
```
- Run multiple evolutions (5+)
- Each with 50-100 genomes × 50 generations = 2500+ hidden entries
- Verify archive DB performance (search, load, list)
```

#### Soak Test 3: Pareto front growth
```
- Track Pareto front size over 50+ generations
- Verify front never shrinks (only new members added or replaced)
- Check memory not unbounded
```

---

## 26. Implementation Roadmap & Effort Estimation

### Phase 1: Metrics Collection & Profiling (Weeks 1-2)
- Backend: add TrainingProfiler struct, instrumentation points in entities.rs
- Frontend: extend TrainingResult, events, store
- Effort: ~60 LOC frontend, ~120 LOC backend
- Blockers: none
- Acceptance: profiler data collected and emitted in events

### Phase 2: Pareto Front Computation (Weeks 2-3)
- Backend: implement dominance relation, pareto_front computation
- Frontend: add Pareto scatter plot visualization
- Effort: ~80 LOC backend, ~150 LOC React/chart.js
- Acceptance: Pareto front displayed, updates per generation

### Phase 3: Device Profiles (Weeks 3-4)
- Backend: DeviceProfile struct, built-in profiles, constraint penalty
- Frontend: device selector UI, show feasible vs infeasible
- Effort: ~100 LOC backend, ~120 LOC frontend
- Acceptance: device profile selectable, fitness adjusted, Pareto highlights feasible

### Phase 4: Genealogy Tracking (Weeks 4-5)
- Backend: GenomeGeneology struct, mutation type logging
- Frontend: genealogy tree component, useEvolutionLoop integration
- Effort: ~150 LOC backend, ~200 LOC React + D3/Cytoscape
- Acceptance: genealogy tree renders, parent-child links correct

### Phase 5: Hidden Library & Weight Persistence (Weeks 5-7)
- Backend: extend library DTO, auto-save during evolution, weight export API
- Frontend: hidden archive page, unhide workflow, export dialog
- Burn integration: model.save() for .safetensors
- Effort: ~250 LOC backend, ~200 LOC frontend
- Blockers: Burn.save() API availability
- Acceptance: genomes auto-saved, weights exportable, archive page functional

### Phase 6: Stopping Criteria (Weeks 7-8)
- Backend: plateau detection, criterion evaluation
- Frontend: criteria configuration panel, progress visualization
- Effort: ~120 LOC backend, ~180 LOC frontend
- Acceptance: user can configure & enable criteria, stop triggered correctly

### Phase 7: UI/UX Polish & Integration (Weeks 8-9)
- Unify all feature UIs into evolution dashboard
- Add tabs, panels, charts
- Effort: ~300 LOC React + styling
- Acceptance: all features integrated in cohesive UI

### Phase 8: Testing & Soak (Weeks 9-10)
- Write all unit + integration + E2E tests
- 72-hour soak test
- Effort: ~500 LOC tests
- Acceptance: pass all tests, no memory leaks, stable under load

**Total Estimated Effort**: ~10-11 weeks, 1 developer
**LOC Estimate**: Backend ~800-1000, Frontend ~1500-1800, Tests ~500-700

---

## 27. Success Criteria & Metrics

### Feature Completeness
- [ ] Performance profiling collected end-to-end (training time, inference latency, peak memory)
- [ ] Pareto front computed, visualized, updated per generation
- [ ] Device profiles selectable, constraints applied to fitness
- [ ] Genealogy tracked, tree viewer functional
- [ ] Hidden library auto-saves all genomes, unhide + reseed works
- [ ] Stopping criteria: all types working (generation, plateau, time, target accuracy)
- [ ] Weight export produces valid .safetensors + metadata

### Performance (Measured)
- Profiling overhead < 5% total training time
- Pareto computation O(N²) completes in < 100ms for N=100
- Hidden library search/list completes in < 500ms for 10K entries
- UI responsiveness maintained (no frame drops) during parallel jobs visualization

### Test Coverage
- Backend unit: > 80% on orthogonal functions
- Integration: 90%+ of workflows tested
- E2E: all major user flows (evolution + export + archive + reseed)
- Soak: 72 hours stable without memory leak

### User Experience
- Settings panel non-overwhelming (organize in tabs/sections)
- Dashboard updates smoothly with 3-5 parallel jobs active
- Pareto front visualization intuitive (2D scatter, highlights non-dominated)
- Export workflow clear (3-4 clicks from genesis selection to weights)


