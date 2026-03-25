# План задач: FSD-совместимая эволюция

Обновлено под текущую структуру проекта после FSD-рефакторинга.

## Ключевые правила структуры

- `pages` композируют экран и роутинг.
- `features` содержат бизнес-логику и feature-UI.
- `widgets` содержат композитные визуализации из нескольких features.
- `shared` содержит общие типы, hooks и утилиты.
- Глобальный evolution state: `src/features/evolution-manager/model/store.ts`.

---

## Phase 1: Infrastructure

| # | Задача | Файл |
|---|--------|------|
| 001 | Backend DTO contracts | [001_backend_dto_contracts.md](./001_backend_dto_contracts.md) |
| 002 | Orchestrator scaffold | [002_orchestrator_module_scaffold.md](./002_orchestrator_module_scaffold.md) |
| 003 | Frontend store extensions | [003_frontend_store_extensions.md](./003_frontend_store_extensions.md) |

---

## Phase 2: Core Features

| # | Задача | Файл |
|---|--------|------|
| 101 | Profiler backend | [101_performance_profiler_backend.md](./101_performance_profiler_backend.md) |
| 102 | Profiler frontend | [102_performance_profiler_frontend.md](./102_performance_profiler_frontend.md) |
| 103 | Pareto backend | [103_pareto_backend.md](./103_pareto_backend.md) |
| 104 | Pareto visualization | [104_pareto_front_visualization_frontend.md](./104_pareto_front_visualization_frontend.md) |
| 105 | Device constraints backend | [105_device_profiles_backend.md](./105_device_profiles_backend.md) |
| 106 | Device constraints frontend | [106_device_profiles_frontend.md](./106_device_profiles_frontend.md) |
| 107 | Genealogy backend | [107_genealogy_tracking_backend.md](./107_genealogy_tracking_backend.md) |
| 108 | Genealogy visualization | [108_genealogy_tree_visualization_frontend.md](./108_genealogy_tree_visualization_frontend.md) |
| 109 | Hidden library backend | [109_hidden_library_backend.md](./109_hidden_library_backend.md) |
| 110 | Weight export | [110_weight_checkpointing_export.md](./110_weight_checkpointing_export.md) |
| 111 | Hidden archive UI | [111_hidden_archive_ui.md](./111_hidden_archive_ui.md) |
| 112 | Stopping backend | [112_stopping_criteria_backend.md](./112_stopping_criteria_backend.md) |
| 113 | Stopping frontend | [113_stopping_criteria_frontend.md](./113_stopping_criteria_frontend.md) |
| 114 | Profiler memory breakdown | [114_profiler_memory_breakdown_backend.md](./114_profiler_memory_breakdown_backend.md) |
| 115 | Device library backend | [115_device_library_backend.md](./115_device_library_backend.md) |
| 116 | Device library frontend | [116_device_library_frontend.md](./116_device_library_frontend.md) |

---

## Phase 3: UI Integration

| # | Задача | Файл |
|---|--------|------|
| 117 | Settings expansion | [117_settings_configuration_ui_expansion.md](./117_settings_configuration_ui_expansion.md) |
| 118 | Evolution dashboard | [118_evolution_progress_dashboard.md](./118_evolution_progress_dashboard.md) |
| 119 | Post-evolution analysis | [119_post_evolution_analysis_panel.md](./119_post_evolution_analysis_panel.md) |

---

## Phase 4: Testing

| # | Задача | Файл |
|---|--------|------|
| 120 | Backend unit tests | [120_backend_unit_tests.md](./120_backend_unit_tests.md) |
| 121 | Frontend unit tests | [121_frontend_unit_tests.md](./121_frontend_unit_tests.md) |
| 122 | Integration tests | [122_integration_tests.md](./122_integration_tests.md) |
| 123 | E2E tests | [123_e2e_tests_tauri_ui.md](./123_e2e_tests_tauri_ui.md) |
| 124 | Soak and stress tests | [124_soak_stress_tests.md](./124_soak_stress_tests.md) |

---

## Критические зависимости

1. Сначала закрыть `001-003`.
2. Затем `101-116`.
3. Затем `117-119`.
4. Затем `120-124`.

Минимальный критический путь:
`001 -> 003 -> 106 -> 116 -> 117 -> 118 -> 119 -> 121 -> 123`

---

## FSD карта внедрения

- Settings page: `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`
- Runtime page: `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- Global evolution store: `src/features/evolution-manager/model/store.ts`
- Dashboard widget: `src/widgets/evolution-dashboard/*`
- Post-analysis widget: `src/widgets/post-evolution-panel/*`
- Device UI: `src/features/evolution-manager/ui/*`
- Hidden archive page: `src/pages/hidden-archive-page/*`
