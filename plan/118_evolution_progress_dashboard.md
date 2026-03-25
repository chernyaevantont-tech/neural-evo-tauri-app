# Задача 118: Evolution Progress Dashboard

**Фаза**: 3 (UI/UX Integration)  
**Сложность**: High  
**Время**: 12 часов  
**Зависимости**: Task 101-116, Task 117  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Построить единый runtime-dashboard эволюции с live-метриками, графиками, jobs panel и вкладками анализа (Pareto, genealogy, stopping criteria, profiler). Dashboard должен быть центральным экраном во время выполнения и отражать device constraints feasibility в реальном времени.

---

## Входные данные

- `src/widgets/*`
- `src/features/evolution-studio/model/useEvolutionLoop.ts`
- `src/features/evolution-studio/ui/GenerationStatsTable.tsx`
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` (текущий runtime экран)
- `src/widgets/genealogy-tree-viewer/*` (T108)
- `src/widgets/pareto-front-visualizer/*` (T104)
- profiler payload (T101/T114)
- stopping criteria payload (T112/T113)
- `plan.md` раздел 24

---

## Пошаговое выполнение

### Шаг 1: Каркас dashboard и layout

Создать/расширить:
- `src/widgets/evolution-dashboard/EvolutionDashboard.tsx`
- `src/widgets/evolution-dashboard/EvolutionDashboard.module.css`

Обновить экспорт:
- `src/widgets/index.ts`

Структура:
- Top overview cards
- Middle charts row
- Right active jobs panel
- Bottom tabs (4 вкладки)
- Control bar

---

### Шаг 2: Top Overview cards

Карточки:
- `Generations Elapsed`
- `Genomes Evaluated`
- `Current Best Fitness`
- `Pareto Front Size`
- `Elapsed Time`
- `ETA`
- `Feasible Ratio` (сколько решений проходит текущие device constraints)

---

### Шаг 3: Middle charts

Добавить графики:
- line: best fitness over generations
- area: average fitness
- line/area: feasible-front size over generations
- optional overlay: constraint pressure (mean violation score)

---

### Шаг 4: Active Jobs panel

Показать параллельные задачи:
- Job ID
- Genome ID
- Stage (`train`/`val`/`test`)
- Progress %
- Status (`running`, `completed`, `failed`, `queued`)
- Duration
- ETA

Добавить фильтры:
- show only running
- show failed

---

### Шаг 5: Bottom tabs integration

Вкладки:
1. `Pareto Front`
2. `Genealogy Tree`
3. `Stopping Criteria`
4. `Performance Metrics`

Для `Performance Metrics` добавить таблицу:
- train duration
- inference latency
- peak active memory
- peak category memory breakdown
- samples/sec

Интеграция без регрессии:
- не удалять существующие `GenerationStatsTable` и `GenomeProfilerModal`
- обернуть их в новые вкладки/секции dashboard или использовать как подкомпоненты

---

### Шаг 6: Control buttons

Добавить контролы:
- `Pause Evolution`
- `Resume Evolution`
- `Stop Evolution`
- `Save Checkpoint`

Состояния:
- disabled при недопустимых переходах
- confirmation для stop
- spinner для async actions

---

### Шаг 7: Ошибки и деградация

Добавить fallback-поведение:
- если часть метрик не пришла, вкладки не падают
- показывать `data unavailable`
- фиксировать ошибки в event log панели

---

## FSD ограничения

- Dashboard относится к `widgets` (композиция нескольких features и shared компонентов).
- Состояние только через `useEvolutionSettingsStore` и `useEvolutionLoop`.
- В page-слое (`EvolutionStudioPage`) только wiring и layout.

---

## Тесты

- Unit tests:
  - selectors для агрегированных metrics
  - форматтеры ETA/time/memory
- Component tests:
  - рендер overview cards
  - переключение вкладок
  - рендер jobs panel и статусов
  - кнопки pause/resume/stop/checkpoint
- Integration UI tests:
  - поток live updates не ломает dashboard
  - feasible ratio меняется при смене device profile

Команда:

```bash
npx vitest run src/widgets/evolution-dashboard
```

---

## Критерии готовности

- ✅ Единый dashboard отображает все ключевые метрики эволюции
- ✅ Tabs интегрируют Pareto/Genealogy/Stopping/Performance
- ✅ Device feasibility видна на уровне overview и графиков
- ✅ Контролы жизненного цикла эволюции работают корректно
- ✅ UI устойчив к частично отсутствующим данным
- ✅ Тесты проходят

---

## Вывод

- Новый/расширенный модуль: `src/widgets/evolution-dashboard/*`
- Интеграция большинства Phase 2 фич в единый live-экран
- Основа для финального анализа в T119
