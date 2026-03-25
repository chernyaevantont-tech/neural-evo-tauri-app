# Задача 106: Device Constraints UI (Frontend)

**Фаза**: 2 (Core Features - Device Constraints)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 105 (backend constraints), Task 003 (frontend store)  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Добавить UI для выбора и настройки ограничений целевого устройства на основе ресурсов: MOPS, RAM, FLASH, latency. Пользователь должен уметь:

- выбрать встроенный профиль
- вручную задать лимиты
- видеть, какие genomes проходят ограничения
- фильтровать Парето-график только по feasible решениям

Примечание: сохранение пользовательских профилей в библиотеку реализуется в T115/T116, но здесь нужно заложить совместимый UI/API-контракт.

---

## Входные данные

- `src/shared/lib/dtos.ts` (DeviceProfile DTO)
- `src/features/evolution-manager/model/store.ts` (актуальные поля: `deviceProfileId`, `isCustomDevice`, `customDeviceParams`, `selectedDeviceProfile`)
- `src/widgets/pareto-front-visualizer/*` (из T104)
- `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx` (текущая форма настроек)
- `plan.md` раздел 20 и 24

---

## Пошаговое выполнение

### Шаг 1: Сопоставить новый UI с текущим store

Использовать существующие поля store и расширить их без создания нового store:

```ts
deviceProfileId: string;
isCustomDevice: boolean;
customDeviceParams?: {
  mops_budget?: number;
  ram_mb: number;
  flash_mb?: number;
  latency_budget_ms: number;
  max_model_size_mb?: number;
};
```

Action-методы:
- переиспользовать существующие `setDeviceProfileId`, `setIsCustomDevice`, `setCustomDeviceParams`, `setSelectedDeviceProfile`
- при необходимости добавить `showOnlyFeasible` и action для фильтра Pareto

---

### Шаг 2: Создать `DeviceProfileSelector` и встроить в текущую Settings Panel

Создать `src/features/evolution-manager/ui/DeviceProfileSelector.tsx`:
- Select built-in profile
- Toggle: "Custom constraints"
- Поля ввода:
  - MOPS budget
  - RAM budget (MB)
  - FLASH budget (MB)
  - Max latency (ms)

UI валидация:
- значения > 0
- числовой формат
- предупреждение, если лимиты слишком низкие (например, почти нулевые)

Отдельно: поддержать одновременно старые поля ограничений (`resourceTargets.flash/ram/macs`) и новые device constraints, чтобы миграция не ломала существующие настройки.

---

### Шаг 3: Интегрировать feasibility в Pareto UI

В `ParetoScatterPlot` передавать:
- `feasibilityByGenomeId`
- `constraintViolationScoreByGenomeId`

Отображать:
- badge `Feasible` / `Not feasible`
- цветовую интенсивность по степени нарушения

---

### Шаг 4: Добавить фильтр "show only feasible"

- Checkbox в панели constraints
- Фильтрация списка и графика перед рендером
- Отдельный счетчик: `feasible / total`

---

### Шаг 5: Подготовить интеграцию с библиотекой устройств

Добавить в компонент extension point:
- callback `onSaveAsTemplate(name)`
- callback `onLoadTemplate(templateId)`

Пока можно заглушить с TODO, реальная реализация в T115/T116.

---

## FSD ограничения

- Размещать UI в `src/features/evolution-manager/ui/`.
- Не импортировать другие feature-модули напрямую.
- Точка подключения в page-слое: `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`.

---

## Тесты

- Unit tests:
  - валидатор input-полей ограничений
  - редьюсер/экшены store
- Component tests:
  - переключение built-in/custom
  - фильтр `showOnlyFeasible`
  - корректная прокидка feasibility props в Pareto компоненты

Команда:

```bash
npx vitest run src/features/evolution-manager/ui/DeviceProfileSelector.test.tsx
```

---

## Критерии готовности

- ✅ Пользователь может задать MOPS/RAM/FLASH/latency ограничения
- ✅ Есть выбор built-in профилей + ручной custom режим
- ✅ Парето визуализация учитывает feasibility
- ✅ Работает фильтрация только feasible genomes
- ✅ Подготовлены extension points под библиотеку устройств
- ✅ Тесты проходят

---

## Вывод

- Новый файл: `src/features/evolution-manager/ui/DeviceProfileSelector.tsx`
- Изменения: store в `src/features/evolution-manager/model/*`, Pareto widgets
- Готовит основу для T115/T116 (сохранение/загрузка пользовательских устройств)
