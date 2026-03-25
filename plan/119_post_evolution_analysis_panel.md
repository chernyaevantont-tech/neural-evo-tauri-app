# Задача 119: Post-Evolution Analysis Panel

**Фаза**: 3 (UI/UX Integration)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 104-116, Task 118  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Реализовать post-run панель для анализа результатов эволюции и экспорта артефактов. Панель должна помогать выбрать финальные genomes с учетом Pareto, genealogy и ограничений устройства.

---

## Входные данные

- `src/widgets/post-evolution-panel/*`
- `src/widgets/pareto-front-visualizer/*`
- `src/widgets/genealogy-tree-viewer/*` (T108)
- hidden archive API/UI (T109/T111)
- weights export flow (T110)
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- `plan.md` раздел 24

---

## Пошаговое выполнение

### Шаг 1: Создать базовый контейнер panel

Создать:
- `src/widgets/post-evolution-panel/PostEvolutionPanel.tsx`
- `src/widgets/post-evolution-panel/PostEvolutionPanel.module.css`

Секции:
- Pareto visualization
- Genome comparison
- Genealogy analysis
- Hidden archive summary
- Export actions

---

### Шаг 2: Pareto visualization with feasibility legend

Добавить:
- 2D scatter
- legend: `feasible` / `infeasible`
- фильтр `show only feasible for selected device`

Под selected device использовать constraints из device settings/device library.

---

### Шаг 3: Detailed comparison (2-3 genomes)

Реализовать multi-select и side-by-side таблицу:
- architecture summary
- accuracy/latency/model size
- training/inference time
- memory breakdown
- lineage depth
- device compatibility ratios (MOPS/RAM/FLASH/latency)

---

### Шаг 4: Genealogy analysis section

Добавить:
- compact family tree
- timeline generation -> fitness
- hover: mutation info
- export lineage as JSON/GraphML

---

### Шаг 5: Hidden library summary

Показывать:
- `N genomes auto-saved`
- quick stats (avg fitness, accuracy range, feasible count)
- кнопка перехода к hidden archive page

Маршрутизация:
- переход на `/hidden-archive` (добавляется в T111)

---

### Шаг 6: Export actions

Кнопки:
- `Download Pareto Front (JSON)`
- `Select & Export Model Weights`
- `Save Evolution Report (PDF)`
- `Continue Evolution`

Для PDF отчета включить секции:
- run config
- top genomes table
- pareto summary
- constraints summary (MOPS/RAM/FLASH/latency)
- stopping reason

Реализация экспорта:
- UI в этом виджете
- вызовы backend/export API через существующие feature-слои

---

## Тесты

- Unit tests:
  - export payload builders (JSON report, PDF data model)
  - comparator logic для side-by-side
- Component tests:
  - выбор genome для сравнения
  - корректный рендер feasible/infeasible legend
  - вызовы export actions
- Integration tests:
  - post-run flow: dashboard -> analysis -> export -> archive

Команда:

```bash
npx vitest run src/widgets/post-evolution-panel
```

---

## Критерии готовности

- ✅ Панель отображает post-run анализ по всем новым feature-доменам
- ✅ Side-by-side сравнение геномов работает
- ✅ Device constraints учитываются в анализе и фильтрах
- ✅ Экспорт JSON/PDF/weights доступен из одной панели
- ✅ Есть прямой переход в hidden archive
- ✅ Тесты проходят

---

## Вывод

- Новый модуль: `src/widgets/post-evolution-panel/*`
- Изменения в `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` (встраивание панели)
- Завершает UX-цепочку: настройка -> live dashboard -> итоговый анализ -> экспорт
