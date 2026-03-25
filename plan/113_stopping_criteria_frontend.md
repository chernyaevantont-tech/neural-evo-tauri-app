# Задача 113: Stopping Criteria (Frontend)

**Фаза**: 2 (Core Features - Control UI)  
**Сложность**: Low  
**Время**: 5 часов  
**Зависимости**: Task 112, Task 003  
**Выполнит**: Frontend разработчик

---

## Описание

Добавить UI конфигурацию и мониторинг критериев остановки:
- pre-run настройка criteria
- live status каждого критерия во время эволюции
- post-run отображение, какой критерий сработал

---

## Входные данные

- API/DTO из T112
- `src/features/evolution-manager/model/store.ts` (`stoppingPolicy`, `currentStoppingProgress`)
- `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx` (pre-run settings)
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx` (runtime визуализация)
- `plan.md` раздел 23 и 24

---

## Пошаговое выполнение

### Шаг 1: Создать `StoppingCriteriaPanel`

Файл:
- `src/features/evolution-manager/ui/StoppingCriteriaPanel.tsx`

Функции:
- список добавленных критериев
- кнопка `Add criterion`
- параметры по типу критерия

---

### Шаг 2: Валидация конфигурации

Проверки:
- `patience > 0`
- `time limit > 0`
- `target accuracy in [0, 1]`
- не более одного `ManualStop` в списке

---

### Шаг 3: Live мониторинг

Во время run показывать:
- progress bar для GenerationLimit
- elapsed timer для TimeLimit
- plateau meter (`patience used / patience total`)
- target accuracy прогресс

---

### Шаг 4: Post-run summary

Показывать:
- stop reason
- triggered criteria
- generation/time at stop

---

### Шаг 5: Интеграция с dashboard

Добавить блок Stopping Criteria в EvolutionDashboard.

До завершения T118 блок должен быть доступен в текущем `EvolutionStudioPage`.

---

## Тесты

- Unit tests для валидатора формы
- Component tests:
  - добавление/удаление критериев
  - корректный рендер параметров
  - рендер stop-reason после завершения

Команда:

```bash
npx vitest run src/features/evolution-manager/ui/StoppingCriteriaPanel.test.tsx
```

---

## Критерии готовности

- ✅ Пользователь может сконфигурировать набор критериев
- ✅ Работает валидация конфигурации
- ✅ Во время run отображается live-прогресс критериев
- ✅ После run видно, что вызвало остановку
- ✅ Тесты проходят

---

## Вывод

- Новый UI компонент: `StoppingCriteriaPanel`
- Изменения в `EvolutionSettingsPanel` и EvolutionDashboard
- Полностью закрывает frontend часть stopping criteria
