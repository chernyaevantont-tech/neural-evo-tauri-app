# План выполнения: Мульти-объектная эволюция и расширенные метрики

**Общее описание**: Проект разбит на 19 независимых задач, выполняемых последовательно разными AI-агентами. Каждая задача полностью самостоятельна и не требует контекста других задач (зависимости указаны явно).

**Общее время**: ~10-11 недель, 1 разработчик (или N агентов параллельно)  
**Всего LOC**: ~2300 (backend 800-1000, frontend 1500-1800, tests 500-700)

---

## ФАЗА 1: Инфраструктура & Контракты (недели 1-2)

| # | Задача | Описание | ⏱️ | 📦 Вывод |
|---|--------|---------|------|--------|
| 1 | [001_backend_dto_contracts](./001_backend_dto_contracts.md) | Define all DTOs (Profiler, Pareto, Genealogy, Device, Stopping) | 4ч | `dtos.rs` +150 LOC |
| 2 | [002_orchestrator_module_scaffold](./002_orchestrator_module_scaffold.md) | Create orchestrator module (scheduler, memory estimator, registry) | 6ч | `orchestrator/` (4 files) +300 LOC |
| 3 | [003_frontend_store_extensions](./003_frontend_store_extensions.md) | Extend Zustand stores + create helper hooks (Pareto, Genealogy, Stopping) | 4ч | types + hooks +300 LOC |

---

## ФАЗА 2: Ядро функций (недели 2-8)

### Metrics & Performance Profiling (неделя 2-3)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 101 | [Performance Profiler Backend](./101_performance_profiler_backend.md) - TrainingProfiler + instrumentation | 8ч | T1 |
| 102 | [Performance Profiler Frontend](./102_performance_profiler_frontend.md) - UI tables, modals, comparison charts | 4ч | T101, T3 |

### Multi-Objective Optimization (неделя 3-4)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 103 | [Pareto Frontend Comp](./103_pareto_backend.md) - dominance, O(N²) frontier | 8ч | T1 |
| 104 | [Pareto Visualization](./104-116_phase2_templates.md#104) - 2D/3D scatter, selection, tooltip | 6ч | T103 |

### Device Profiles System (неделя 4-5)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 105 | [Device Profiles Backend](./104-116_phase2_templates.md#105) - 9 built-in + custom, constraint penalty | 6ч | T1, T103 |
| 106 | [Device Profiles Frontend](./104-116_phase2_templates.md#106) - selector, filtering, badges | 4ч | T105, T104 |

### Genealogy Tracking (неделя 5-6)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 107 | [Genealogy Backend](./104-116_phase2_templates.md#107) - parent_ids, mutations, cycle detection | 8ч | T1 |
| 108 | [Genealogy Visualization](./104-116_phase2_templates.md#108) - tree viewer, D3/Cytoscape | 8ч | T107 |

### Hidden Library & Weight Persistence (неделя 6-7)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 109 | [Hidden Library Backend](./104-116_phase2_templates.md#109) - auto-save, is_hidden flag | 8ч | T1, T107, T101 |
| 110 | [Weight Checkpointing](./104-116_phase2_templates.md#110) - export .safetensors + metadata | 8ч | T101, T109 |
| 111 | [Hidden Archive UI](./104-116_phase2_templates.md#111) - archive page, search, bulk ops, unhide | 6ч | T109, T110 |

### Stopping Criteria System (неделя 7-8)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|---------|
| 112 | [Stopping Criteria Backend](./104-116_phase2_templates.md#112) - plateau, time, accuracy, policy | 6ч | T1 |
| 113 | [Stopping Criteria Frontend](./104-116_phase2_templates.md#113) - config panel, progress viz | 5ч | T112 |

---

## ФАЗА 3: UI/UX Интеграция (неделя 8-9)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|----------|
| 117 | [Settings UI Expansion](./117-119_phase3_ui_templates.md#117) - Device, MOO, Stopping panels | 6ч | T105, T112 |
| 118 | [Evolution Dashboard](./117-119_phase3_ui_templates.md#118) - Unified dashboard + 4 tabs (Pareto, genealogy, criteria, metrics) | 10ч | T101-113 |
| 119 | [Post-Evolution Analysis](./117-119_phase3_ui_templates.md#119) - Pareto viz, comparison, genealogy, export | 6ч | T104-111 |

---

## ФАЗА 4: Тестирование (неделя 9-10)

| # | Задача | ⏱️ | 📚 Deps |
|---|--------|------|----------|
| 120 | [Backend Unit Tests](./120-124_phase4_testing_templates.md#120) - Profiler, Pareto, Device, Genealogy, Stopping (80% coverage) | 10ч | T101-113 |
| 121 | [Frontend Unit Tests](./120-124_phase4_testing_templates.md#121) - Hooks, components, rendering (80% coverage) | 8ч | T102, T104-116 |
| 122 | [Integration Tests](./120-124_phase4_testing_templates.md#122) - 6 scenarios (MOO, device, archive, genealogy, stopping, parallel) | 12ч | All Phase 2 |
| 123 | [E2E Tests](./120-124_phase4_testing_templates.md#123) - 5 workflows (evolution, pause/resume, archive, constraints, export) | 10ч | All Phases 1-3 |
| 124 | [Soak & Stress Tests](./120-124_phase4_testing_templates.md#124) - 72h stability, memory, performance | 4h setup + 72h run | All |

---

## Матрица зависимостей

```
001 ─────────┬─────────┬─────────┐
             │         │         │
         002 │         │         │
             ▼         │         │
         103 ◄─────────┼─────────┤
         104 ◄────────┐│         │
         105 ◄────────┼┼─┐       │
         106 ◄────────┼┼─┤       │
         107 ◄────────┼┼─┤       │
         108 ◄────────┼┼─┤       │
         109 ◄────────┼│ │       │
         110 ◄────────┼┼─┤       │
         112 ◄────────┼┼─┤       │
         113 ◄────────┼┼─┤       │
        003 ◄─────────┴┴─┴───────┘
        201 ◄──────────────────────────┐
        202 ◄─────────────────────────┐│
        203 ◄────────────────────────┐││
        30X ◄───────────────────────┐│││
```

---

## Модель выполнения

**Вариант 1: Последовательно (1 разработчик)**
- Фаза 1: недели 1-2 (3 задачи, 14ч)
- Фаза 2: недели 2-7 (13 задач, расписаны выше)
- Фаза 3: неделя 8-9 (3 задачи, 22ч)
- Фаза 4: неделя 9-10 (5 задач, тестирование)
- **Итого**: ~11 недель

**Вариант 2: Параллельно (N агентов)**
- Фаза 1: дождаться завершения (недели 1-2)
- Фаза 2A (profiler, Pareto): недели 2-4, 2 агента одновременно
- Фаза 2B (device, genealogy, hidden library): недели 4-7, 3 агента одновременно
- Фаза 2C (stopping criteria): неделя 7-8, 1 агент
- Фаза 3: только после 2, недели 8-9, 3 агента (UI параллельно)
- Фаза 4: недели 9-10, 5 агентов (тесты параллельно)
- **Итого**: ~6-7 недель

---

## Критерии готовности каждой задачи

Каждая задача должна завершиться с:
- ✅ Код скомпилирован / lint passed
- ✅ Tests written + passing (если применимо)
- ✅ Acceptance criteria met
- ✅ PR/commits ready для merge
- ✅ No merge conflicts intro main branch

---

## Как запустить задачу

1. Откройте файл задачи (e.g., `plan/001_backend_dto_contracts.md`)
2. Прочитайте раздел "## Описание" и "## Входные данные"
3. Следуйте пошаговым инструкциям в разделе "## Пошаговое выполнение"
4. Запустите тесты из раздела "## Тесты"
5. Проверьте "## Критерии готовности"
6. Если всё OK → можно переходить к следующей задаче

---

## Файловая структура после выполнения

```
src-tauri/src/
├── lib.rs (обновления)
├── dtos.rs (расширение -- задача 1)
├── profiler.rs (NEW -- задача 4)
├── pareto.rs (NEW -- задача 6)
├── device_profiles.rs (NEW -- задача 8)
├── genealogy.rs (NEW -- задача 10)
├── stopping_criteria.rs (NEW -- задача 15)
└── orchestrator/ (NEW -- задача 2)
    ├── mod.rs
    ├── scheduler.rs
    └── memory_estimator.rs

src/
├── entities/canvas-genome/model/
│   └── store.ts (расширение -- задача 3)
├── features/
│   ├── evolution-studio/model/store.ts (расширение)
│   ├── genealogy-viewer/ (NEW -- задача 11)
│   └── [UI компоненты ниже]
├── widgets/
│   ├── pareto-front-visualizer/ (NEW -- задача 7)
│   ├── genealogy-tree-viewer/ (NEW -- задача 11)
│   ├── evolution-dashboard/ (NEW -- задача 18)
│   └── post-evolution-panel/ (NEW -- задача 19)
├── pages/
│   └── hidden-archive-page/ (NEW -- задача 14)
└── [тесты в параллельни directories с .test.ts]
```

---

**Дата создания**: 2026-03-24  
**Версия плана**: 2.0 (структурирован по задачам)  
**Статус**: Ready for execution
