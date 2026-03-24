# Задача 120-124: Phase 4 Tasks - Testing & Soak

**Фаза**: 4 (Testing & Validation)  
**Зависимости**: All Phases 1-3

---

## Задача 120: Backend Unit Tests

**Сложность**: Medium | **Время**: 10ч | **Зависимости**: Tasks 101-115

### Описание
Написать comprehensive unit tests для всех backend модулей:

#### Coverage targets:
- `profiler.rs`: timing accuracy, memory tracking, throughput calculation (~15 tests)
- `pareto.rs`: dominance relation, frontier computation, edge cases (~12 tests)
- `device_profiles.rs`: constraint penalties, built-in profiles validation (~10 tests)
- `genealogy.rs`: parent-child links, mutation serialization, cycle detection (~12 tests)
- `stopping_criteria.rs`: plateau detection, time limit, criteria evaluation (~10 tests)
- `orchestrator/*`: scheduler, memory estimation, run registry (~15 tests)

### Структура
```
src-tauri/src/
├── profiler.rs (tests inline)
├── pareto.rs (tests inline)
├── device_profiles.rs (tests inline)
├── genealogy.rs (tests inline)
├── stopping_criteria.rs (tests inline)
└── orchestrator/
    ├── mod.rs (tests inline)
    ├── scheduler.rs (tests inline)
    ├── memory_estimator.rs (tests inline)
    └── run_registry.rs (tests inline)
```

### Запуск
```bash
cargo test --lib profiler
cargo test --lib pareto
cargo test --lib orchestrator
# ... все тесты
```

### Критерии
- ✅ > 80% code coverage по orthogonal функциям
- ✅ Все edge cases протестированы (пустые inputs, overflow, boundary conditions)
- ✅ Tests pass на CI/CD
- ✅ No flaky tests

### Вывод
- ~500 LOC файлов с #[cfg(test)]
- Coverage report

---

## Задача 121: Frontend Unit Tests

**Сложность**: Medium | **Время**: 8ч | **Зависимости**: Tasks 102, 104-116

### Описание
Написать unit tests для frontend компонент + hooks:

#### Coverage targets:
- `useParetoTracking`: computeParetoFront, isDominated (~8 tests)
- `useGenealogy`: buildAncestralChain, hasCycles, cycle detection (8 tests)
- `useStoppingCriteria`: checkGenerationLimit, checkPlateau, checkTimeLimit (~8 tests)
- `ComparisonCharts`: render, data mapping, tooltip (~6 tests)
- `GenomeProfilerModal`: display profiler data, formatting (~6 tests)
- `ParetoScatterPlot`: scatter data correctness, selection (~8 tests)

### Инструменты
- Vitest (existing setup)
- React Testing Library
- jsdom environment

### Структура
```
src/
├── shared/hooks/
│   ├── useParetoTracking.test.ts
│   ├── useGenealogy.test.ts
│   └── useStoppingCriteria.test.ts
├── widgets/
│   ├── pareto-front-visualizer/
│   │   └── ParetoScatterPlot.test.tsx
│   ├── genome-comparison/
│   │   └── ComparisonCharts.test.tsx
└── features/
    ├── evolution-studio/ui/
    │   └── GenomeProfilerModal.test.tsx
```

### Запуск
```bash
npm run test -- src/shared/hooks/
npm run test -- src/widgets/
npm run test -- src/features/evolution-studio/
```

### Критерии
- ✅ > 80% coverage
- ✅ All tests pass
- ✅ No console errors/warnings

### Вывод
- ~400 LOC test файлов

---

## Задача 122: Integration Tests

**Сложность**: High | **Время**: 12ч | **Зависимости**: All Phase 2

### Описание
Integration tests для 6 основных сценариев:

#### Scenario 1: Multi-Objective Evolution E2E
- Start MOO evolution
- Run 5 generations
- Verify Pareto front computed after each gen
- Verify objectives tracked (accuracy, latency, size)
- Check profiler data collected

#### Scenario 2: Device-Aware Fitness
- Set device profile (embedded)
- Evolve population
- Verify fitness penalties applied
- Check Pareto front only feasible genomes

#### Scenario 3: Auto-save + Hidden Library
- Run evolution 2 gens
- Verify all genomes auto-saved as hidden
- Export best genome with weights
- Check .safetensors + metadata.json files

#### Scenario 4: Genealogy Tracking
- Start with random population
- Gen 1: mutations + crossover
- Verify parent_ids set correctly
- Build genealogy tree, check no cycles

#### Scenario 5: Stopping Criteria Triggered
- Set criteria: FitnessPlateau (patience=3), TimeLimit (30s)
- Run evolution
- Verify stop triggered by correct criterion
- Check run status updated

#### Scenario 6: Parallel Training Profiling
- Set max_parallel_jobs = 3
- Enqueue 10 genomes
- Verify max 3 active simultaneously
- Verify timing + memory profiling on all
- Check VRAM reservation cycle

### Инструменты
- Tokio for async tests
- Custom test fixtures
- Mock Burn models (lightweight)

### Структура
```
src-tauri/tests/
├── integration_moo_evolution.rs
├── integration_device_aware.rs
├── integration_auto_save_export.rs
├── integration_genealogy.rs
├── integration_stopping_criteria.rs
└── integration_parallel_profiling.rs
```

### Запуск
```bash
cargo test --test integration_*
```

### Критерии
- ✅ All 6 scenarios pass
- ✅ No flaky tests
- ✅ Reasonable timing (< 60s per scenario)

### Вывод
- ~600 LOC test файлов
- Integration test suite

---

## Задача 123: E2E Tests (Tauri + UI)

**Сложность**: High | **Время**: 10ч | **Зависимости**: All Phases 1-3

### Описание
E2E tests для 5 основных user workflows:

#### Test 1: Complete MOO Evolution Workflow
- Load dataset
- Enter settings: pop=20, gens=3, MOO=true, device=mobile
- Click Start
- Monitor progress
- Verify Pareto front updates
- Stop evolution
- View post-analysis
- Select genome, export weights

#### Test 2: Stopping Criteria Pause/Resume
- Configure: FitnessPlateau (patience=2)
- Start evolution
- Pause after gen 2
- Resume, run gen 3-4
- Verify stop triggered on plateau

#### Test 3: Hidden Archive Unhide & Reseed
- Run first evolution
- Check archive (N hidden genomes)
- Open archive page
- Search + unhide best genome
- Start new evolution with unhidden as seed

#### Test 4: Device Profile Constraints
- Select embedded device
- Run evolution
- View Pareto front
- Verify: feasible (green), infeasible (red)
- Export: only feasible on front

#### Test 5: Weight Export Workflow
- Run evolution
- View Pareto front
- Select top genome
- Click "Export Model Weights"
- Dialog: choose folder, confirm
- Verify zip contains: genome.json, .safetensors, metadata.json

### Инструменты
- Tauri WebDriver или Playwright
- VS Code Test Explorer integration

### Структура
```
e2e_tests/
├── moo_evolution.e2e.ts
├── stopping_criteria.e2e.ts
├── hidden_archive.e2e.ts
├── device_constraints.e2e.ts
└── weight_export.e2e.ts
```

### Запуск
```bash
npm run test:e2e
```

### Критерии
- ✅ All 5 workflows pass
- ✅ No flaky UI interactions
- ✅ Reasonable timing (< 2 min per test)

### Вывод
- ~500 LOC E2E test файлов

---

## Задача 124: Soak & Stress Tests

**Сложность**: Medium | **Время**: 4ч setup + 72ч run

### Описание
Long-running soak tests для stability validation:

#### Soak Test 1: Long-running Evolution
- Population: 50 genomes
- Generations: 100+
- Duration: 24-48 hours (depending on compute)
- Targets: memory stability, no leak, consistent performance

#### Soak Test 2: Archive Growth
- Run 5 evolutions sequentially
- Each: 50-100 genomes × 50 generations = 2500+ hidden entries
- Verify: archive DB performance (search, list, load)
- No slowdown over time

#### Soak Test 3: Pareto Front Stability
- Track front size over 50+ generations
- Verify: front never shrinks (monotonic increase or stable)
- Check: memory not unbounded
- Profile: no memory leaks

### Метрики
- Peak memory usage
- Memory leak detection
- CPU utilization
- Responsiveness (UI frame rate)
- Event processing latency

### Инструменты
- Memory profiler (valgrind, heaptrack if Linux)
- Perf counters
- Custom logging + analysis scripts

### Структура
```
soak_tests/
├── runner.rs (main soak orchestrator)
├── long_evolution.rs
├── archive_growth.rs
└── pareto_stability.rs
```

### Запуск
```bash
cargo build --release
./run_soak_tests.sh  # 72-hour test
```

### Критерии
- ✅ No panics or crashes over 72h
- ✅ Memory delta < 100MB (acceptable leak rate)
- ✅ 99th percentile latency stable
- ✅ No performance degradation over time

### Вывод
- Soak test setup + scripts
- Performance report (HTML / PDF)
- Recommendations для улучшений

