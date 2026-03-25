# QUICKSTART: Запуск задач после FSD-рефакторинга

## Что изменилось

План полностью синхронизирован с текущей структурой проекта:
- backend задачи в `src-tauri/src/*`
- frontend задачи строго по слоям `pages/features/widgets/shared`
- все тестовые задачи вынесены в отдельные файлы `120-124`

---

## С чего начать

1. Открой [00_INDEX.md](./00_INDEX.md).
2. Выполни `001`, `002`, `003`.
3. Перейди к Phase 2 (`101-116`).
4. После завершения core-фич выполняй `117-119`.
5. Закрой quality gate задачами `120-124`.

---

## Где внедрять frontend изменения

- Настройки: `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`
- Runtime экран: `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- Feature state: `src/features/evolution-manager/model/store.ts`
- Feature UI: `src/features/evolution-manager/ui/*`
- Композитные визуализации: `src/widgets/*`
- Роутинг: `src/app/App.tsx`

---

## Правило для целевого устройства

Во всех релевантных задачах device constraints задаются как:
- MOPS
- RAM MB
- FLASH MB
- Max latency ms

И поддерживается библиотека пользовательских device templates (сохранение между сессиями).

---

## Быстрый контроль готовности

- Есть реализация всех задач `001-124` по файлам из `plan/`.
- Нет ссылок на устаревшие pre-FSD пути.
- Frontend задачи указывают реальные точки интеграции в `pages/features/widgets`.
- Тестовые задачи покрывают `features`, `widgets`, `pages` и backend API.
