# Neural Evo Tauri App

Десктопное приложение для визуального проектирования архитектур нейросетей и их эволюционного поиска (NAS).

Стек: React + TypeScript + Tauri 2 + Rust + Burn 0.20.

## Актуальный статус (март 2026)

### Что уже работает
- Визуальный редактор графов нейросетей (sandbox) с мутациями, кроссовером, валидацией совместимости и автоподстройкой форм тензоров.
- Dataset Manager с профилями датасетов, сканированием, кэшированием, валидацией и предпросмотром CSV.
- Evolution Studio с асинхронным циклом эволюции, live-метриками, журналом, остановкой по запросу и политиками остановки.
- Multi-objective оптимизация и Pareto front (accuracy/latency/model size и др.).
- Ограничения устройств (device profiles): feasibility-check, penalty, библиотека шаблонов устройств (CRUD + import/export).
- Genealogy tracking (история происхождения геномов: founder/mutation/crossover).
- Genome Library и Hidden Archive (автосохранение результатов, восстановление, удаление).
- Экспорт генома с весами и проверка наличия кэша весов.

### Что частично/экспериментально
- Zero-cost proxies (ускоренные прокси-оценки) присутствуют на backend и частично в UI, но интеграция в основной цикл эволюции еще развивается.
- Time-series сценарии ориентированы в первую очередь на Conv1D-подход; RNN/LSTM/GRU требуют отдельной доработки в рамках текущих ограничений Burn-экосистемы и проектного бэклога.

### Что еще в плане
- Phase 4 тестирования: frontend unit tests, integration, E2E (Tauri UI), soak/stress (см. `plan/120-124_phase4_testing_templates.md`).

## Основные возможности

### 1) Визуальный редактор архитектур
- Узлы: Input, Dense, Conv2D, Pooling, Flatten, BatchNorm, LayerNorm, Dropout, Dropout2D, GaussianNoise, Add, Concat2D, Output.
- Действия: добавление, редактирование, копирование, удаление узлов; создание/удаление связей; сохранение/загрузка геномов `.evog`.
- Проверка совместимости связей и распространение shape через граф.

### 2) Эволюционные операторы
- Структурные мутации: `AddNode`, `RemoveNode`, `RemoveSubgraph`.
- Кроссовер геномов с извлечением/вставкой подграфов.
- Контроль разрастания: лимиты размера графа и парсимония (penalty за сложность).

### 3) Эволюционная оркестрация и аналитика
- Асинхронная пакетная оценка популяции в Rust backend.
- Stratified split для категориальных задач (train/val/test).
- Profiling: время, пропускная способность, разбивка памяти.
- Pareto-анализ и визуализация компромиссов качества/ресурсов.
- Пост-эволюционный анализ и скрытый архив результатов.

## Карта приложения

Роуты в приложении:
- `/` - Home
- `/sandbox` - редактор архитектур
- `/dataset-manager` - менеджер датасетов
- `/evolution-studio` - запуск и мониторинг эволюции
- `/genome-library` - библиотека геномов
- `/hidden-archive` - скрытый архив эволюционных результатов

## Подробный гайд пользователя

### Шаг 1. Создайте или загрузите базовую архитектуру
1. Откройте `Architecture Sandbox` (`/sandbox`).
2. Добавьте входной и выходной узлы, затем промежуточные слои.
3. Соединяйте узлы через `Shift + Click` (source -> target).
4. Настройте параметры слоев через контекстное меню узла.
5. Сохраните удачный геном в `.evog` (или в библиотеку).

### Шаг 2. Подготовьте датасет
1. Перейдите в `Dataset Manager` (`/dataset-manager`).
2. Создайте профиль датасета (Folder или CSV).
3. Для каждого stream задайте роль (`Input`/`Target`) и locator.
4. Нажмите `Scan` для проверки структуры и извлечения статистики.
5. При необходимости выполните `Cache` (особенно для image-потоков).
6. Нажмите `Validate` и убедитесь, что профиль готов к эволюции.

### Шаг 3. Запустите эволюцию
1. Перейдите в `Evolution Studio` (`/evolution-studio`).
2. Выберите dataset profile.
3. Настройте population size, поколения, мутации, ограничения устройства, stopping criteria.
4. Добавьте seed-геномы (из canvas/библиотеки) или используйте random initialization.
5. Нажмите `Start Evolution`.

Во время выполнения:
- отслеживайте live-метрики и логи,
- смотрите динамику fitness,
- анализируйте Pareto-множество,
- при необходимости используйте `Pause`/`Stop`.

### Шаг 4. Проанализируйте и сохраните результаты
1. Просмотрите `Hall of Fame`, Pareto-участников и пост-анализ на дашборде.
2. Откройте `Genome Library` для сохраненных решений.
3. При необходимости экспортируйте геном вместе с весами.
4. Используйте `Hidden Archive` для анализа автоматически сохраненных результатов и lineage.

## Полный разбор настроек Evolution Manager

Ниже описаны все настройки, доступные в панели Evolution Settings на странице Evolution Studio.

Важно: часть параметров (например train/val/test split) берется из профиля датасета в Dataset Manager, а не из этой панели.

### 1) Service actions

- Apply
	- Нормализует objective weights.
	- Валидирует конфигурацию через buildEvolutionRunConfig.
	- Сохраняет конфиг как last used в localStorage.
- Save preset
	- Сохраняет текущие настройки как preset в localStorage.
- Load last used config
	- Загружает последний сохраненный конфиг из localStorage.

### 2) Objectives

- Optimization mode
	- Single-Objective.
	- Multi-Objective (Pareto).
- Secondary objectives (для multi-objective)
	- latency.
	- model_size.
	- train_time.
- Use weighted aggregation
	- Включает агрегацию по весам, иначе используется Pareto-логика без weighted-score.
- Weights sliders
	- accuracy, latency, model_size, train_time.
	- Диапазон каждого веса: 0..1, шаг 0.05.
- Normalize weights
	- Приводит сумму весов к 1.0.

Значения по умолчанию:
- mobjEnabled: false.
- secondaryObjectives: latency + model_size.
- objectiveWeightsEnabled: true.
- objectiveWeights: accuracy 0.5, latency 0.2, model_size 0.2, train_time 0.1.

### 3) Crossover Strategies

Можно включить одну или несколько стратегий одновременно:
- subgraph-insertion.
- subgraph-replacement.
- neat-style.
- multi-point.

Поведение:
- при кроссовере выбирается случайная стратегия из включенных;
- последнюю активную стратегию отключить нельзя.

По умолчанию:
- selectedCrossovers: subgraph-insertion.

### 4) Mutation Probabilities

Параметры мутаций (0..1, шаг 0.05):
- params.
- addNode.
- removeNode.
- removeSubgraph.
- addSkipConnection.
- changeLayerType.

По умолчанию:
- params 0.6.
- addNode 0.2.
- removeNode 0.1.
- removeSubgraph 0.05.
- addSkipConnection 0.3.
- changeLayerType 0.1.

Adaptive Mutation:
- переключатель useAdaptiveMutation.
- параметр adaptiveTargetNodes (целевое число узлов, по умолчанию 20).
- при включении вручную не редактируются addNode/removeNode/removeSubgraph: они вычисляются динамически от текущего размера генома.

### 5) Bloat Control

- Max Nodes Limit
	- useMaxNodesLimit: включает жесткий лимит узлов.
	- maxNodesLimit: значение лимита.
- Parsimony Pressure
	- useParsimonyPressure.
	- parsimonyAlpha.
	- Итоговая fitness-коррекция: adjustedFitness = baseFitness - alpha * nodeCount.

По умолчанию:
- useMaxNodesLimit: false.
- maxNodesLimit: 30.
- useParsimonyPressure: false.
- parsimonyAlpha: 0.01.

### 6) Resource Awareness

- useResourceAwareFitness.
- resourceTargets:
	- flash (bytes).
	- ram (bytes).
	- macs.

Как влияет:
- При превышении лимитов к fitness применяется penalty на основе относительного превышения flash/ram/macs.

По умолчанию:
- useResourceAwareFitness: false.
- flash: 1048576 (1 MB).
- ram: 262144 (256 KB).
- macs: 1000000.

### 7) Device Targeting

Параметры доступны в секции Device Targeting:

- Built-in profile
	- embedded-mcu.
	- edge-tiny.
	- mobile-low-end.
	- mobile-mid-range.
	- laptop-cpu.
- Custom constraints
	- mops_budget.
	- ram_mb.
	- flash_mb.
	- latency_budget_ms.
- Show only feasible
	- фильтрует показ решений по device feasibility.
- Device templates
	- save/apply/update/duplicate/delete.
	- import/export библиотеки шаблонов.

Дополнительно:
- Секция показывает estimated parallelism на основе RAM-ограничения и memorySafetyMargin.
- Изменение device constraints синхронизирует resourceTargets для resource-aware fitness.

По умолчанию:
- deviceProfileId: default-device (после инициализации выбирается первый built-in профиль).
- isCustomDevice: false.
- showOnlyFeasible: false.

### 8) Random Initialization

- useRandomInitialization.
- randomInitRatio (0..100, шаг 5).

Как работает:
- Если включено и есть seed-геномы: часть популяции генерируется случайно, часть создается от seed.
- Если seed нет: может быть полностью random initialization.

По умолчанию:
- useRandomInitialization: false.
- randomInitRatio: 30.

### 9) Training Parameters

- batchSize
	- UI диапазон: 1..512.
	- По умолчанию: 32.
- evalEpochs
	- UI диапазон: 1..100.
	- По умолчанию: 1.
- datasetPercent
	- UI диапазон: 1..100%.
	- По умолчанию: 100%.
- populationSize
	- Ограничивается в store: 4..200.
	- По умолчанию: 20.
- Max Generations
	- useMaxGenerations.
	- maxGenerations (минимум 1).
	- По умолчанию: выключено, maxGenerations 100.

Важно:
- train/val/test split в evaluate_population берется из выбранного dataset profile (поле split), а не из этой панели.

### 10) Stopping Criteria

Настраивается через отдельную панель:

- Policy type
	- any (OR): остановка, когда выполнен любой критерий.
	- all (AND): остановка, когда выполнены все критерии.
- Типы критериев:
	- GenerationLimit: max_generations > 0.
	- FitnessPlateau:
		- monitor: best_fitness | pareto_coverage | population_avg.
		- patience_generations > 0.
		- improvement_threshold >= 0.
	- TimeLimit: max_seconds > 0.
	- TargetAccuracy: threshold от 0 до 1.
	- ManualStop.

Ограничения валидации:
- критерий должен быть минимум один;
- ManualStop может быть только один.

По умолчанию:
- stoppingPolicy: criteria = [ManualStop], policy_type = any.

### 11) Advanced Performance

Секция скрыта по умолчанию (Show advanced settings):

- profilingEnabled.
- memorySafetyMarginMb (минимум 0).
- estimatorSafetyFactor (минимум 1).
- memoryMode: estimate | runtime | hybrid.

По умолчанию:
- profilingEnabled: false.
- memorySafetyMarginMb: 128.
- estimatorSafetyFactor: 1.1.
- memoryMode: hybrid.

### 12) Zero-Cost Proxy Evaluation

- useZeroCostProxies.
- zeroCostStrategy:
	- two-stage.
	- early-stopping.
- fastPassThreshold (0..1, шаг 0.05).
- partialTrainingEpochs (1..50 в UI; в store ограничение 1..100).

Как влияет на обучение:
- Для каждого генома сначала считается zero-cost score.
- В зависимости от score геном может:
	- быть пропущен,
	- обучаться частично,
	- обучаться полностью.
- Если zero-cost выключен, используется evalEpochs для всех геномов.

По умолчанию:
- useZeroCostProxies: false.
- zeroCostStrategy: two-stage.
- fastPassThreshold: 0.6.
- partialTrainingEpochs: 20.

### 13) Runtime/analysis toggles, связанные с менеджером

Эти флаги хранятся в том же evolution settings store и влияют на отображение/сбор данных:
- genealogyTrackingEnabled (по умолчанию true).
- autoSaveToHiddenLibrary (по умолчанию false).
- showOnlyFeasible (по умолчанию false).

## Рекомендованные стартовые пресеты

### Быстрый smoke-тест
- populationSize: 8-12.
- evalEpochs: 1.
- datasetPercent: 10-25.
- useZeroCostProxies: on, strategy two-stage.
- useMaxGenerations: on, maxGenerations: 5-10.

### Баланс качества/времени
- populationSize: 20-40.
- evalEpochs: 2-5.
- datasetPercent: 40-70.
- useParsimonyPressure: on, alpha 0.005-0.02.
- multi-objective: on (latency + model_size).

### Поиск под edge-device
- mobjEnabled: true.
- secondaryObjectives: latency + model_size (+ train_time опционально).
- включить Device Targeting и задать реальные бюджеты MOPS/RAM/FLASH/latency.
- showOnlyFeasible: on.
- useResourceAwareFitness: on и синхронизировать resourceTargets с device constraints.

## Управление canvas

| Действие | Управление |
|---|---|
| Панорамирование | Правая кнопка мыши + drag |
| Зум | Колесо мыши (с центром под курсором) |
| Выбор | Левый клик |
| Соединение узлов | `Shift + Click` |
| Контекстное меню | Правая кнопка на узле/связи/геноме |

## Быстрый старт для разработчика

### Требования
- Node.js 18+
- Rust stable (через rustup)
- Tauri CLI (`@tauri-apps/cli`, уже в devDependencies)

### Установка

```bash
npm install
```

### Запуск

```bash
# Frontend (Vite)
npm run dev

# Desktop (Tauri)
npm run tauri dev
```

### Сборка

```bash
npm run build
npm run tauri build
```

### Тесты

```bash
# Frontend tests (Vitest)
npm run test

# One-shot запуск конкретного теста
npx vitest run src/app/App.integration.test.tsx

# Backend tests (из папки src-tauri)
cd src-tauri
cargo test --lib
```

## Архитектура

Проект следует Feature-Sliced Design (FSD):
- `pages/` - экраны и композиция,
- `features/` - прикладная логика и feature-компоненты,
- `entities/` - доменные сущности,
- `widgets/` - крупные композиционные блоки,
- `shared/` - инфраструктура и общие утилиты.

Ключевые frontend-папки:
- `src/pages/`
- `src/features/`
- `src/widgets/`
- `src/entities/canvas-genome/`
- `src/entities/canvas-state/`

Ключевые backend-модули:
- `src-tauri/src/entities.rs` - компиляция/выполнение графа, train/eval
- `src-tauri/src/data_loader.rs` - загрузка данных и батчинг
- `src-tauri/src/profiler.rs` - профайлинг
- `src-tauri/src/pareto.rs` - multi-objective и Pareto
- `src-tauri/src/device_profiles.rs`, `device_library.rs`
- `src-tauri/src/genealogy.rs`
- `src-tauri/src/stopping_criteria.rs`
- `src-tauri/src/weight_io.rs`
- `src-tauri/src/orchestrator/`

## Tauri API (актуально)

На данный момент в `generate_handler!` зарегистрировано 42 команды, включая:
- Genome I/O: `save_genome`, `load_genome`
- Dataset: `scan_dataset`, `cache_dataset`, `validate_dataset_profile`, `save_dataset_profiles`, `load_dataset_profiles`, `preview_csv`, `pick_folder`
- Evolution: `evaluate_population`, `stop_evolution`
- Library/Archive: `list_library_genomes`, `save_to_library`, `load_library_genome`, `delete_from_library`, `list_hidden_library`, `unhide_genome`, `delete_hidden_genome`
- Weights: `export_genome_with_weights`, `has_cached_weights`
- Pareto/Zero-cost: `compute_pareto_front`, `compute_zero_cost_score`
- Devices: `get_device_profiles`, `validate_genome_for_device`, `apply_device_penalty`, `list_device_templates`, `create_device_template`, `update_device_template`, `delete_device_template`, `duplicate_device_template`, `export_device_library`, `import_device_library`
- Genealogy: `register_founder`, `register_mutation`, `register_crossover`, `get_genealogy`, `get_ancestors`, `get_descendants`
- Stopping: `validate_stopping_criteria`, `generate_stopping_preview`

## Текущий статус тестирования

- Backend unit tests (Task 120): 112/115 passed (97%), покрыты ключевые модули профайлера, Pareto, device constraints, genealogy, orchestrator, data loader.
- Frontend/integration/E2E/soak тесты (Tasks 121-124): запланированы и частично в работе по roadmap.

## Ограничения и known issues

- Автоматические shape-adapter'ы могут раздувать архитектуру; компенсируется лимитами и penalty за сложность.
- Speciation-механизм в стиле NEAT пока не внедрен.
- Для time-series основная рабочая стратегия сейчас - Conv1D-подобные архитектуры; RNN/LSTM/GRU требуют отдельного этапа реализации.
- Часть продвинутых оптимизаций (например, полная интеграция zero-cost стратегий в основной runtime) находится в развитии.

## Дополнительная документация

- Общая архитектура backend: `BACKEND_ARCHITECTURE_ANALYSIS.md`
- Правила canvas/consistency: `NETWORK-CANVAS.md`
- Справка по архитектуре нод: `NODE_ARCHITECTURE_REFERENCE.md`
- Текущее улучшение эволюции и crossover: `EVOLUTION.MD`
- План рефакторинга данных: `DATASET_REFACTORING_PLAN.md`
- План работ и roadmap: `plan/00_INDEX.md`
- Отчет по backend тестам: `TASK_120_COMPLETION_REPORT.md`