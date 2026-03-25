# Задача 117: Settings & Configuration UI Expansion

**Фаза**: 3 (UI/UX Integration)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 104, Task 105, Task 106, Task 112, Task 113, Task 115, Task 116  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Расширить панель конфигурации эволюции, чтобы пользователь мог централизованно настроить:
- режим оптимизации (single/multi objective)
- целевое устройство через ресурсные ограничения
- библиотеку сохраненных устройств
- критерии остановки
- расширенные параметры профилирования и оценки ресурсов

Ключевой UX-требование: пользователь не должен повторно вбивать MOPS/RAM/FLASH/latency при каждом запуске, если профиль уже сохранен в библиотеке устройств.

---

## Входные данные

- `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx` (текущая монолитная панель)
- `src/features/evolution-manager/ui/DeviceProfileSelector.tsx` (T106)
- `src/features/evolution-manager/ui/DeviceLibraryManager.tsx` (T116)
- `src/features/evolution-manager/ui/StoppingCriteriaPanel.tsx` (T113)
- `src/features/evolution-manager/model/store.ts` (существующие поля ресурсов и advanced tracking)
- `src/shared/lib/dtos.ts`
- `plan.md` раздел 20, 23, 24

---

## Пошаговое выполнение

### Шаг 1: Рефакторинг Settings panel на секции

Обновить `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`:
- выделить 4 секции в отдельные компоненты
- собрать их в общий layout с единым submit/apply behavior

Новые секции:
- `ObjectivesSection`
- `DeviceTargetingSection`
- `StoppingCriteriaSection`
- `AdvancedPerformanceSection`

Рекомендуемое размещение секций:
- `src/features/evolution-manager/ui/settings-sections/*`

---

### Шаг 2: Performance & Objectives section

Добавить:
- переключатель `Single-Objective` / `Multi-Objective (Pareto)`
- чекбоксы вторичных целей (`latency`, `model_size`, `train_time`)
- опциональные sliders приоритетов

Валидация:
- в multi-objective минимум 2 цели
- веса нормализуются в `sum = 1` (если режим с весами активен)

---

### Шаг 3: Device Targeting section

Добавить в UI:
- dropdown встроенных профилей
- переключение на custom constraints
- поля:
  - `MOPS budget`
  - `RAM budget (MB)`
  - `FLASH budget (MB)`
  - `Max latency (ms)`

  Совместимость с текущим store:
  - мигрировать от текущих `resourceTargets.flash/ram/macs` к device-параметрам без потери обратной совместимости
  - поддержать mapping `MACs -> MOPS` и нормализовать единицы измерения в UI

Интегрировать библиотеку устройств:
- блок `Saved Device Templates`
- кнопки `Apply`, `Save current as template`, `Edit`, `Delete`

Отображать:
- `Estimated parallelism for selected constraints: N`
- предупреждение, если constraints слишком жесткие и feasible решений мало

---

### Шаг 4: Stopping Criteria section

Встроить `StoppingCriteriaPanel` в settings и унифицировать стиль с остальными секциями.

Добавить:
- переключатель policy `any/all`
- summary-бейджи по активным критериям

---

### Шаг 5: Advanced Performance section (collapsed)

Добавить сворачиваемый блок:
- `Safety margin (MB)`
- `Estimator safety factor`
- `Profiling enabled`
- `Memory mode`: `estimate/runtime/hybrid`

Параметры должны быть сериализуемы в конфиг запуска эволюции.

---

### Шаг 6: Сохранение/восстановление конфигурации

Добавить:
- `Save preset` (локальный пресет всей формы)
- `Load last used config`

Это отдельный слой от device library: device library хранит только профиль устройства, preset хранит всю панель настроек.

---

## FSD ограничения

- Page-слой: только композиция (`EvolutionSettingsPanel.tsx`).
- UI-секции и actions: `features/evolution-manager`.
- Не создавать отдельный settings-store: использовать текущий `useEvolutionSettingsStore`.

---

## Тесты

- Unit tests:
  - нормализация objective weights
  - валидатор device constraints
  - сериализация form state -> run config
- Component tests:
  - переключение single/multi objective
  - применение device template автозаполняет поля
  - сохранение device template из custom constraints
  - policy `any/all` сохраняется в конфиге

Команда:

```bash
npx vitest run src/pages/evolution-studio-page
```

---

## Критерии готовности

- ✅ Настройки разделены на 4 явные секции
- ✅ Device constraints задаются через MOPS/RAM/FLASH/latency
- ✅ Библиотека устройств интегрирована напрямую в settings
- ✅ Stopping criteria и advanced perf полностью конфигурируются
- ✅ Конфиг запуска формируется детерминированно и валидируется
- ✅ Тесты проходят

---

## Вывод

- Основной файл: `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`
- Новые секции: `src/features/evolution-manager/ui/*Section*.tsx`
- Интеграция с T116 (device library manager) и T113 (stopping criteria)
