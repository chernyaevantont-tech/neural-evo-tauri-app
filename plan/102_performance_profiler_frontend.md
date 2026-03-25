# Задача 102: Performance Profiler Frontend

**Фаза**: 2 (Core Features)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 101, Task 003  
**Выполнит**: Frontend разработчик

---

## Описание

Встроить profiler-метрики в текущую FSD-структуру UI:
- таблица поколений
- детали генома
- runtime визуализация на странице эволюции

---

## Входные данные

- `src/features/evolution-studio/ui/GenerationStatsTable.tsx`
- `src/features/evolution-studio/ui/GenomeProfilerModal.tsx`
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- `src/features/evolution-manager/model/store.ts`

---

## Пошаговое выполнение

### Шаг 1: Обновить таблицу поколений

В `GenerationStatsTable` добавить поля:
- training time
- inference latency
- peak memory
- throughput

### Шаг 2: Уточнить profiler modal

В `GenomeProfilerModal` показать breakdown по train/val/test и memory categories.

### Шаг 3: Интегрировать в runtime page

В `EvolutionStudioPage` связать live snapshot с profiler UI без создания нового global state.

### Шаг 4: Подготовить переиспользование в dashboard

Сделать компоненты пригодными для переиспользования в `widgets/evolution-dashboard` (T118).

---

## FSD ограничения

- Локальный UI profiler: слой `features/evolution-studio/ui`.
- Композитный dashboard: слой `widgets` (в T118).
- Page слой только композирует.

---

## Тесты

```bash
npx vitest run src/features/evolution-studio/ui
```

Проверить:
- корректный рендер profiler-полей
- graceful fallback при частично отсутствующих данных

---

## Критерии готовности

-  Профайлер отображается в текущем UI эволюции
-  Нет регрессий в `EvolutionStudioPage`
-  Компоненты готовы для интеграции в T118
-  Тесты проходят

---

## Вывод

- Изменения: `src/features/evolution-studio/ui/*`, `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- Подготовка к unified dashboard

