# Задача 111: Hidden Archive UI

**Фаза**: 2 (Core Features - Persistence UI)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 109, Task 110, Task 003  
**Выполнит**: Frontend разработчик

---

## Описание

Создать страницу управления скрытым архивом геномов:
- таблица со всеми hidden entries
- поиск/фильтры по поколениям и метрикам
- batch-операции (unhide/delete/export)
- detail modal с genealogy + profiler

---

## Входные данные

- API T109/T110
- `src/app/App.tsx` (актуальные маршруты)
- `src/pages/genome-library-page/*` (существующая библиотека)
- `src/features/genome-library/*` (переиспользование существующих UI-компонентов)
- `plan.md` раздел 22, 24

---

## Пошаговое выполнение

### Шаг 1: Создать страницу архива

Создать:
- `src/pages/hidden-archive-page/HiddenArchivePage.tsx`
- `src/pages/hidden-archive-page/index.ts`

---

### Шаг 2: Таблица и фильтры

Колонки:
- genome_id
- generation
- accuracy
- latency
- model_size
- created_at
- parents_count

Фильтры:
- generation range
- accuracy range
- latency range
- search by genome id

---

### Шаг 3: Batch actions

Реализовать:
- `Unhide selected`
- `Delete selected`
- `Export selected`

Показывать confirmation dialog и результат операции.

---

### Шаг 4: Detail modal

Показывать:
- objectives
- profiler breakdown
- lineage summary
- quick actions: open genealogy, export weights

---

### Шаг 5: Интеграция в маршрутизацию

Добавить роут и пункт меню в app shell.

Минимум:
- добавить route `/hidden-archive` в `src/app/App.tsx`
- добавить переход из `src/pages/genome-library-page/GenomeLibraryPage.tsx`

---

## Тесты

- Unit tests фильтров/сортировки
- Component tests:
  - selection + batch action
  - открытие detail modal
  - обработка пустого состояния

Команда:

```bash
npx vitest run src/pages/hidden-archive-page
```

---

## Критерии готовности

- ✅ Hidden archive page доступна из UI
- ✅ Работают поиск, фильтры, сортировка
- ✅ Работают batch-операции
- ✅ Detail modal показывает profiler + lineage
- ✅ Тесты проходят

---

## FSD ограничения

- Страница: слой `pages` (`src/pages/hidden-archive-page/*`).
- Доменная логика и state: переиспользовать `features/genome-library` и `features/evolution-manager`.
- Не дублировать CRUD, если аналог уже есть в `genome-library` feature.

---

## Вывод

- Новая страница: `src/pages/hidden-archive-page/*`
- Изменения: роутинг, меню, library интеграция
- Основа для post-evolution анализа в T119
