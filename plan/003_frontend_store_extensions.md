# Задача 003: Frontend Store Extensions

**Фаза**: 1 (Infrastructure)  
**Сложность**: Medium  
**Время**: 5 часов  
**Зависимости**: Task 001  
**Выполнит**: Frontend разработчик

---

## Описание

Привести frontend state-контракты к текущей FSD-архитектуре и подготовить единый источник состояния для новых фич.

Ключевой принцип: не создавать новые глобальные store, расширять существующий `useEvolutionSettingsStore`.

---

## Входные данные

- `src/features/evolution-manager/model/store.ts`
- `src/shared/lib/dtos.ts`
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`

---

## Пошаговое выполнение

### Шаг 1: Синхронизировать frontend DTO mirror

В `src/shared/lib/dtos.ts` добавить/обновить типы:
- profiler
- pareto
- device constraints and templates
- genealogy
- stopping

### Шаг 2: Расширить `useEvolutionSettingsStore`

Добавить runtime/setting поля для:
- pareto history
- genealogy tree
- generation profiling stats
- stopping progress
- hidden archive counters
- device template selection

### Шаг 3: Подготовить селекторы и helper actions

В `features/evolution-manager/model` добавить helper-функции без дублирования бизнес-логики в `pages`.

### Шаг 4: Проверить совместимость текущего UI

`EvolutionSettingsPanel` и `EvolutionStudioPage` должны продолжать работать без регрессий.

---

## FSD ограничения

- Глобальный state: только `features/evolution-manager/model/store.ts`.
- UI-композиция: `pages/*`.
- Визуальные композиции: `widgets/*`.
- Не делать `feature -> feature` импортов.

---

## Тесты

```bash
npx vitest run src/features/evolution-manager src/shared/lib
```

Проверить:
- reducer/actions корректны
- начальные значения не ломают существующие страницы

---

## Критерии готовности

-  Store покрывает все новые домены
-  DTO и store согласованы
-  Нет регрессий в текущих страницах
-  Тесты проходят

---

## Вывод

- Изменения: `src/features/evolution-manager/model/store.ts`, `src/shared/lib/dtos.ts`
- Основа для всех frontend задач Phase 2-4

