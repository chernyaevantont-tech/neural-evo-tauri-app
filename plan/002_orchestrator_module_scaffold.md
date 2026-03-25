# Задача 002: Orchestrator Module Scaffold

**Фаза**: 1 (Infrastructure)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 001  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Создать каркас backend-оркестратора для queue-first параллельного обучения:
- scheduler
- memory estimator
- run registry
- lifecycle команд

FSD-примечание: backend-only, но API/события должны быть пригодны для потребления текущими фронтенд-слоями (`features/evolution-studio`, `features/evolution-manager`, `widgets/evolution-dashboard`).

---

## Входные данные

- `src-tauri/src/lib.rs`
- `src-tauri/src/entities.rs`
- `src-tauri/src/orchestrator/`

---

## Пошаговое выполнение

### Шаг 1: Создать модульную структуру

- `src-tauri/src/orchestrator/mod.rs`
- `src-tauri/src/orchestrator/scheduler.rs`
- `src-tauri/src/orchestrator/memory_estimator.rs`
- `src-tauri/src/orchestrator/run_registry.rs`

### Шаг 2: Реализовать queue-first scheduling

- enqueue jobs
- admission по памяти
- управление active/queued/completed/failed

### Шаг 3: Интегрировать memory safety

Учитывать safety margin и estimator safety factor из конфигурации.

### Шаг 4: Экспортировать команды и статусы run

Подготовить API для фронтенда:
- status polling
- progress метрики
- pause/resume/stop

---

## Тесты

```bash
cargo test --lib orchestrator
```

Проверить:
- корректность admission
- отсутствие гонок в registry
- корректные state transitions

---

## Критерии готовности

-  Каркас orchestrator готов и компилируется
-  Queue-first pipeline работает
-  Run lifecycle покрыт базовыми тестами
-  API пригоден для задач T118/T123

---

## Вывод

- Изменения: `src-tauri/src/orchestrator/*`, `src-tauri/src/lib.rs`
- Основа для производственного режима эволюции

