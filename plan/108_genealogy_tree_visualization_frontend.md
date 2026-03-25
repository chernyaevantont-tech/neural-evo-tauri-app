# Задача 108: Genealogy Tree Visualization (Frontend)

**Фаза**: 2 (Core Features - Genealogy)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 107 (backend genealogy), Task 003 (frontend types/hooks)  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Построить интерактивный просмотр genealogic tree для evolved genomes:
- узлы: genomes
- ребра: parent -> child
- подписи ребер: mutation type
- фильтры по generation, fitness, Pareto-only
- клик по узлу открывает подробности и сравнение с родителями

---

## Входные данные

- API T107 (`get_genealogy`, `get_ancestors`, `get_descendants`)
- `src/features/evolution-manager/model/store.ts` (`genealogyTree` как runtime state)
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- `src/widgets/*` для нового композитного visualizer
- `plan.md` раздел 21 и 24

---

## Пошаговое выполнение

### Шаг 1: Создать виджет genealogy-tree-viewer

Создать:
- `src/widgets/genealogy-tree-viewer/GenealogicTreeView.tsx`
- `src/widgets/genealogy-tree-viewer/GenealogyFilters.tsx`
- `src/widgets/genealogy-tree-viewer/index.ts`

Допустимо вынести преобразование данных в:
- `src/shared/hooks/useGenealogyGraph.ts`

---

### Шаг 2: Реализовать layout дерева

Использовать D3.js или Cytoscape.js:
- Y-axis: generation
- X-axis: позиция внутри поколения
- edge labels: mutation type

Обязательные функции:
- zoom/pan
- fit-to-screen
- center-on-selected-node

---

### Шаг 3: Реализовать фильтры

Фильтрация:
- поколение (`generation min/max`)
- fitness range
- Pareto only
- show ancestors depth (0..N)

---

### Шаг 4: Node details panel

При клике на node:
- genome id, generation
- parents list
- mutation type
- objectives/profiler summary
- кнопка "Compare with parent"

---

### Шаг 5: Интеграция в dashboard

Добавить отдельную вкладку `Genealogy` в evolution dashboard/post-analysis.

Также добавить промежуточную интеграцию в текущий `EvolutionStudioPage`, пока T118 не завершен.

---

## Тесты

- Unit tests:
  - transform raw genealogy -> graph nodes/edges
  - фильтры по generation/fitness
- Component tests:
  - click на node открывает details
  - изменение фильтров меняет рендер

Команда:

```bash
npx vitest run src/widgets/genealogy-tree-viewer
```

---

## Критерии готовности

- ✅ Есть интерактивный tree-view по lineage данным
- ✅ Поддерживаются zoom/pan + фильтры
- ✅ Отображаются labels мутаций и связи родителей
- ✅ По клику открывается карточка genome
- ✅ Вкладка интегрирована в dashboard
- ✅ Тесты проходят

---

## Вывод

- Новый виджет: `src/widgets/genealogy-tree-viewer/*`
- Изменения: `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` и dashboard UI
- Используется в T118/T119
