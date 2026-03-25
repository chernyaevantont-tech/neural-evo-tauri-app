# Задача 116: Device Library Manager UI (Frontend)

**Фаза**: 2 (Core Features - Device Constraints UX)  
**Сложность**: Medium  
**Время**: 7 часов  
**Зависимости**: Task 106, Task 115, Task 003  
**Выполнит**: Frontend разработчик (React/TypeScript)

---

## Описание

Добавить UI-менеджер библиотеки устройств, где пользователь может:
- сохранять текущие ограничения как шаблон устройства
- выбирать сохраненные устройства в один клик
- редактировать/удалять/дублировать профили
- импортировать/экспортировать библиотеку

Это закрывает основной UX-поток: не вводить MOPS/RAM/FLASH/latency каждый раз заново.

---

## Входные данные

- Backend API T115
- `src/features/evolution-manager/ui/DeviceProfileSelector.tsx` (из T106)
- `src/features/evolution-manager/model/store.ts`
- `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`
- `src/shared/lib/dtos.ts`
- `plan.md` раздел 24 (UI расширение)

---

## Пошаговое выполнение

### Шаг 1: Создать менеджер устройств

Создать:
- `src/features/evolution-manager/ui/DeviceLibraryManager.tsx`
- `src/features/evolution-manager/model/useDeviceLibrary.ts`

---

### Шаг 2: Реализовать список профилей

Показывать таблицу:
- name
- MOPS
- RAM MB
- FLASH MB
- max latency ms
- updated at

Actions на строку:
- `Apply`
- `Edit`
- `Duplicate`
- `Delete`

---

### Шаг 3: Save current constraints as template

В `DeviceProfileSelector` добавить кнопку:
- `Save as device template`

Открывать форму:
- `name`
- `notes`
- `tags`

---

### Шаг 4: Import/Export UI

Добавить кнопки:
- `Import library`
- `Export library`

Показать результат операции (кол-во импортированных/обновленных записей).

---

### Шаг 5: Интегрировать в pre-evolution settings

Встроить DeviceLibraryManager рядом с DeviceProfileSelector:
- быстрый выбор сохраненного устройства
- автозаполнение лимитов
- пометка активного шаблона

Обязательная точка встраивания:
- `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`

---

## Тесты

- Unit tests hooks:
  - list/create/update/delete flows
  - обработка API ошибок
- Component tests:
  - save template from current constraints
  - apply template обновляет selector
  - import/export кнопки вызывают нужные actions

Команда:

```bash
npx vitest run src/features/evolution-manager
```

---

## Критерии готовности

- ✅ Пользователь может сохранить устройство в библиотеку
- ✅ Сохраненные устройства доступны между сессиями (через backend T115)
- ✅ Можно редактировать/дублировать/удалять шаблоны
- ✅ Можно импортировать и экспортировать библиотеку
- ✅ Device selector использует библиотеку в один клик
- ✅ Тесты проходят

---

## FSD ограничения

- Компоненты остаются в `src/features/evolution-manager/ui/`.
- Логика загрузки/мутаций шаблонов в `src/features/evolution-manager/model/`.
- Page-слой только композирует UI, без бизнес-логики.

---

## Вывод

- Новые файлы: `DeviceLibraryManager.tsx`, `useDeviceLibrary.ts`
- Изменения: `DeviceProfileSelector.tsx`, settings UI
- Полностью закрывает UX-требование по библиотеке целевых устройств
