# Задача 112: Stopping Criteria (Backend)

**Фаза**: 2 (Core Features - Control Logic)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 001  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Добавить гибкую систему критериев остановки эволюции с политикой `any/all`:
- `GenerationLimit`
- `FitnessPlateau`
- `TimeLimit`
- `TargetAccuracy`
- `ManualStop`

---

## Входные данные

- `src-tauri/src/dtos.rs`
- `src-tauri/src/entities.rs` (evolution loop)
- `src-tauri/src/lib.rs`
- `plan.md` раздел 23

---

## Пошаговое выполнение

### Шаг 1: Создать модуль `stopping_criteria.rs`

Определить:
- enum критериев
- состояние трекера (best history, elapsed time)
- evaluator

---

### Шаг 2: Реализовать evaluator

```rust
pub fn check_stopping_criteria(
    criteria: &[StoppingCriterion],
    state: &EvolutionProgressState,
    policy: StoppingPolicy,
) -> StoppingDecision
```

`StoppingDecision` должен содержать:
- `should_stop`
- `triggered_criteria`
- `reason_message`

---

### Шаг 3: Plateau detection

Реализовать логику:
- скользящее окно `patience` поколений
- stop, если улучшение `< min_delta`

---

### Шаг 4: Интеграция в loop

На каждой итерации поколения:
- обновлять state
- вызывать evaluator
- если stop: завершать run с reason

---

### Шаг 5: Tauri API

Добавить:
- `validate_stopping_config`
- `evaluate_stopping_preview` (опционально)

---

## Тесты

- Unit:
  - каждый критерий отдельно
  - policy `any` и `all`
  - multiple triggers
- Integration:
  - эволюция завершается по plateau/time limit/target accuracy

Команда:

```bash
cargo test --lib stopping_criteria
```

---

## Критерии готовности

- ✅ Все 5 критериев реализованы
- ✅ Policy any/all работает корректно
- ✅ Trigger reason передается в результат run
- ✅ Интеграция в evolution loop завершена
- ✅ Тесты проходят

---

## Вывод

- Новый файл: `src-tauri/src/stopping_criteria.rs`
- Изменения: `src-tauri/src/entities.rs`, `src-tauri/src/lib.rs`
- База для frontend панели T113
