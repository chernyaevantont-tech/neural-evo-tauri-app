# Задача 104-116: Phase 2 Tasks Templates

**Фаза**: 2 (Core Features)

---

## Задача 104: Pareto Front Visualization (Frontend)

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Task 103

### Описание
Создать компоненты для 2D/3D визуализации Парето-фронта:
- 2D scatter plot (Accuracy vs Latency, размер = model_size)
- Highlight non-dominated genomes (bolded, unique color)
- Mouse-over tooltip с metrics
- Selection для "Use as seed" или "Export"

### Вывод
- Компонент `src/widgets/pareto-front-visualizer/ParetoScatterPlot.tsx`
- Компонент `src/widgets/pareto-front-visualizer/ParetoSelector.tsx`
- Hooks для Pareto query + dominance check
- Integration в EvolutionDashboard

---

## Задача 105: Device Profiles Backend

**Сложность**: Low | **Время**: 6ч | **Зависимости**: Task 001, Task 103

### Описание
Реализовать device profile system:
- 9 built-in profiles (embedded, mobile, laptop, cloud)
- Custom profile configuration
- Device-aware fitness penalty: quadratic для constraint violations
- API: `get_device_profiles()`, `validate_genome_for_device()`

### Вывод
- Модуль `src-tauri/src/device_profiles.rs`
- Constants для built-in profiles
- Constraint penalty функция

---

## Задача 106: Device Profiles Frontend

**Сложность**: Low | **Время**: 4ч | **Зависимости**: Task 105, Task 003

### Описание
UI для device profile selection:
- Dropdown с 9 built-in profiles
- Custom fields (RAM, VRAM, latency_budget, max_model_size)
- Show "all constraints met" badge на Парето-фронте
- Filter: show only feasible genomes

### Вывод
- Компонент `src/features/evolution-manager/ui/DeviceProfileSelector.tsx`
- Updated Парето visualizer с constraint filtering
- Store extension (Task 003 already included)

---

## Задача 107: Genealogy Tracking (Backend)

**Сложность**: Medium | **Время**: 8ч | **Зависимости**: Task 001

### Описание
Реализовать отслеживание генеалогии:
- Store parent_ids + mutation_type в каждом геноме
- Логировать mutations: AddNode, RemoveNode, Crossover и т.д.
- Validate no cycles в family tree
- API: `get_genealogy(genome_id)` -> ancestral chain

### Вывод
- Модуль `src-tauri/src/genealogy.rs`
- Integration в evolution loop (useEvolutionLoop + store)

---

## Задача 108: Genealogy Tree Visualization (Frontend)

**Сложность**: Medium | **Время**: 8ч | **Зависимости**: Task 107

### Описание
Компонент для визуализации family tree:
- Tree layout: generation 0 -> N на Y-axis, времён на X-axis
- Nodes = genomes, edges = parent-child с mutation labels
- Click node -> genome details + compare with parents
- Filter: show only Pareto, or by fitness range

### Вывод
- Компонент `src/features/genealogy-viewer/ui/GenealogicTreeView.tsx`
- D3.js или Cytoscape.js integration
- Hooks для tree traversal

---

## Задача 109: Hidden Library (Backend)

**Сложность**: Medium | **Время**: 8ч | **Зависимости**: Task 001, Task 107, Task 101

### Описание
Auto-save all evolved genomes как hidden entries в библиотеку:
- During generation: save каждый trained genome
- Extended DTO: `is_hidden=true`, `source_generation`, `parent_genomes`, `fitness_metrics`, `profiler_data`
- API: `list_hidden_library()`, `unhide_genome(id)`, `mark_as_hidden()`

### Вывод
- Backend API в `genome-library/` модуле
- Auto-save integration в training loop

---

## Задача 110: Weight Checkpointing & Export

**Сложность**: High | **Время**: 8ч | **Зависимости**: Task 101, Task 109

### Описание
Сохранение весов trained моделей:
- After evolution: user selects genome from Pareto
- Re-train или load weights if cached
- Export to `.safetensors` + `metadata.json` (accuracy, loss, timestamp, profiler)
- Backend API: `export_genome_with_weights(genome_id, output_path)`

### Вывод
- Backend API в entities.rs / weight saver module
- Burn `.save()` integration
- Frontend export dialog

---

## Задача 111: Hidden Archive UI

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Task 109, Task 110

### Описание
Новая страница для управления hidden архивом:
- Searchable table: genome_id, generation, accuracy, latency, parents, created_at
- Filters: generation range, accuracy/latency/size ranges, device profile
- Bulk ops: unhide all, delete all, export batch
- Detail modal: full genealogy chain, profiler breakdown

### Вывод
- Новая страница `src/pages/hidden-archive-page/HiddenArchivePage.tsx`
- Table component с фильтрацией и поиском
- Detail modal

---

## Задача 112: Stopping Criteria (Backend)

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Task 001

### Описание
Реализовать stopping criteria logic:
- `GenerationLimit`, `FitnessPlateau`, `TimeLimit`, `TargetAccuracy`, `ManualStop`
- Plateau detection: compare best fitness over patience generations
- Policy: "any" (stop if ANY criterion met) или "all" (stop if ALL)
- API: `check_stopping_criteria()` per generation

### Вывод
- Модуль `src-tauri/src/stopping_criteria.rs`
- Integration в evolution loop (check per generation)

---

## Задача 113: Stopping Criteria (Frontend)

**Сложность**: Low | **Время**: 5ч | **Зависимости**: Task 112

### Описание
UI для stopping criteria configuration и monitoring:
- Pre-evolution panel: Add criteria, configure parameters
- During evolution: progress bars per criterion, plateau patience meter, time/gen limits
- Post-evolution: show which criterion triggered stop

### Вывод
- Компонент `src/features/evolution-manager/ui/StoppingCriteriaPanel.tsx`
- Progress visualization component
- Updated EvolutionDashboard tab

---

## Задача 114: Profiler Memory Breakdown (Backend)

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Task 101, Task 002

### Описание
Довести backend-профайлер до полноценного memory breakdown без заглушек:
- Убрать `0.0` placeholders для `peak_model_params_mb`, `peak_gradient_mb`, `peak_optim_state_mb`, `peak_activation_mb`
- Добавить deterministic estimator по tensor shapes + dtype (fallback)
- По возможности подключить runtime peak-источник из backend allocator/device telemetry
- Согласовать сбор метрик в training/validation/test проходах

### Пошаговое выполнение
1. Расширить `src-tauri/src/profiler.rs` методами для записи memory breakdown категорий
2. Добавить helper модуль (или секцию) для оценки памяти:
	 - `estimate_model_params_mb(model)`
	 - `estimate_gradients_mb(model)`
	 - `estimate_optimizer_state_mb(optimizer, model)`
	 - `estimate_activations_mb(batch_shapes, node_shapes)`
3. Интегрировать estimation points в `run_eval_pass` и `run_validation_pass`
4. Если runtime telemetry доступна, обновлять `peak_active_memory_mb` и category peaks на каждом batch
5. Добавить feature flag/strategy:
	 - `profiling.memory_mode = "estimate" | "runtime" | "hybrid"`
6. Обновить сериализацию и payload проверками (должны приходить non-zero значения на нетривиальных моделях)

### Тесты
- Unit tests в `profiler.rs`:
	- проверка формул memory estimates
	- проверка monotonic peak updates
	- проверка fallback поведения при недоступном runtime telemetry
- Integration smoke test:
	- маленькая модель (Dense-only)
	- убедиться, что все `peak_*_mb` > 0 (кроме edge-case tiny tensor)

### Критерии готовности
- ✅ `TrainingProfiler` заполняет все `peak_*_mb` категории без постоянных заглушек
- ✅ Значения категорий детерминированны в режиме `estimate`
- ✅ В режиме `hybrid` runtime peak корректно перекрывает estimation при наличии telemetry
- ✅ Тесты проходят (`cargo test --lib profiler` + integration smoke)

### Вывод
- Изменения: `src-tauri/src/profiler.rs`, `src-tauri/src/entities.rs`, опционально `src-tauri/src/orchestrator/memory_estimator.rs`
- Результат: корректные memory breakdown метрики для UI задач T102/T118/T119 и тестов T120/T122

---

## Задача 115-116: Reserved для дополнительных features

Если потребуются дополнительные backend функции или UI, они займут эти слоты.

