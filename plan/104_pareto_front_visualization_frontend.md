# Задача 104: Pareto Front Visualization (Frontend)

**Фаза**: 2 (Core Features - Multi-Objective)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 103 (Pareto backend), Task 003 (frontend DTO/hooks)  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Реализовать визуализацию Парето-фронта в UI эволюции. Поддержать 2D scatter (Accuracy vs Latency), выделение non-dominated genomes, тултипы с ключевыми метриками и действия над точкой (выбрать как seed, экспортировать, открыть детали).

Важно: визуализация должна уметь показывать и дальнейшие ограничения целевых устройств (задачи T105/T106/T115/T116), поэтому архитектура компонента должна быть расширяемой для overlay badges и фильтрации по feasibility.

---

## Входные данные

- `src/shared/lib/dtos.ts` (типы `GenomeObjectives`, `GenerationParetoFront`)
- `src/features/evolution-manager/model/store.ts` (single source of truth: `paretoHistory`, `currentParetoFront`)
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` (текущая точка встраивания)
- `src/widgets/` (место для нового композитного виджета)
- `plan.md` раздел 19 и 24

---

## Пошаговое выполнение

### Шаг 1: Создать структуру виджета

Создать директорию:
- `src/widgets/pareto-front-visualizer/`

Создать файлы:
- `ParetoScatterPlot.tsx`
- `ParetoSelector.tsx`
- `index.ts`
- `pareto-front-visualizer.module.css`

---

### Шаг 2: Реализовать 2D scatter с выделением фронта

В `ParetoScatterPlot.tsx`:

```tsx
import { Scatter } from "react-chartjs-2";
import type { GenerationParetoFront, GenomeObjectives } from "@/shared/lib/dtos";

type Props = {
  pareto: GenerationParetoFront;
  selectedGenomeId?: string;
  onSelectGenome: (genomeId: string) => void;
};

export function ParetoScatterPlot({ pareto, selectedGenomeId, onSelectGenome }: Props) {
  const points = pareto.all_genomes;
  const front = new Set(pareto.frontier_genome_ids);

  return (
    <div>
      {/* Scatter chart with 2 datasets: dominated + frontier */}
    </div>
  );
}
```

Требования к стилю точек:
- Non-dominated: увеличенный радиус, контрастный цвет, обводка
- Dominated: меньшая непрозрачность
- Selected point: отдельный outline

---

### Шаг 3: Добавить tooltip и действия

В tooltip отображать:
- `genome_id`
- `accuracy`
- `inference_latency_ms`
- `model_size_mb`
- `train_time_ms` (если есть profiler)
- `device_feasible` (если уже рассчитано)

В `ParetoSelector.tsx` добавить кнопки:
- `Use as seed`
- `Open details`
- `Export selected`

---

### Шаг 4: Интегрировать в текущий EvolutionStudioPage

Добавить блок Pareto в:
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`

Если в рамках T118 создается отдельный dashboard-widget, переиспользовать этот же компонент без дублирования логики.

Обновить экспорт:
- `src/widgets/index.ts` (добавить экспорт нового виджета)

Поведение:
- Автоматическое обновление при каждом поколении
- Возможность переключения: "current generation" / "global front"

---

### Шаг 5: Подготовить API для device overlays

Добавить в пропсы визуализатора опциональные поля:

```ts
feasibilityByGenomeId?: Record<string, boolean>;
constraintViolationScoreByGenomeId?: Record<string, number>;
```

Это позволит в T106/T116 показывать устройства и фильтрацию без переработки базового графика.

---

## Тесты

- Unit (`vitest`) для:
  - корректного рендера frontier/dominated наборов
  - корректного выбора точки по click
  - корректного отображения tooltip значений
- Component tests (`@testing-library/react`):
  - `onSelectGenome` вызывается с правильным `genome_id`
  - selected state отображается визуально

Команда:

```bash
npx vitest run src/widgets/pareto-front-visualizer
```

---

## Критерии готовности

- ✅ Созданы компоненты `ParetoScatterPlot` и `ParetoSelector`
- ✅ Non-dominated точки визуально выделены
- ✅ Tooltip и действия по выбранной точке работают
- ✅ Компонент интегрирован в dashboard
- ✅ Есть API-хуки для device feasibility overlay
- ✅ Тесты проходят

---

## FSD ограничения

- Компонент относится к слою `widgets` (композитный UI).
- Не импортировать напрямую из других `features/*`.
- Источник состояния: только `useEvolutionSettingsStore` из `src/features/evolution-manager/model/store.ts`.

---

## Вывод

- Новые файлы: `src/widgets/pareto-front-visualizer/*`
- Изменения: `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` и `src/widgets/index.ts`
- Готовит базу для T106 (device filtering), T118, T119
