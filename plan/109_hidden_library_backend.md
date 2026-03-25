# Задача 109: Hidden Library (Backend)

**Фаза**: 2 (Core Features - Persistence)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001, Task 107, Task 101  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Добавить автоматическое сохранение всех обученных genomes в скрытую библиотеку (hidden archive). Каждый entry должен содержать lineage, objectives и profiler-метрики для последующего анализа и повторного использования.

---

## Входные данные

- `src-tauri/src/dtos.rs`
- `src-tauri/src/entities.rs`
- существующая логика genome library (`src/features/genome-library/*` + backend API)
- `plan.md` раздел 22

---

## Пошаговое выполнение

### Шаг 1: Расширить DTO библиотеки

Добавить в entry:
- `is_hidden: bool`
- `source_generation: u32`
- `parent_genomes: Vec<String>`
- `fitness_metrics`
- `profiler_data`
- `created_at_unix_ms`

---

### Шаг 2: Реализовать hidden storage слой

В backend добавить функции:
- `save_hidden_genome(entry)`
- `list_hidden_library(query)`
- `unhide_genome(genome_id)`
- `delete_hidden_genome(genome_id)`

---

### Шаг 3: Интегрировать autosave в evolution loop

На этапе завершения обучения genome:
- собрать objectives + profiler + genealogy
- сохранить entry с `is_hidden = true`
- логировать ошибки сохранения без падения всей эволюции

---

### Шаг 4: Добавить Tauri commands

В `lib.rs`:
- `list_hidden_library`
- `unhide_genome`
- `delete_hidden_genome`

---

### Шаг 5: Добавить индексы/поиск

Поддержать фильтры:
- by generation range
- by accuracy/latency/model_size
- by parent genome id
- by date range

---

## Тесты

- Unit:
  - serialize/deserialize hidden entry
  - save/list/unhide/delete
- Integration:
  - короткий evolution run сохраняет hidden entries
  - фильтры возвращают корректные подмножества

Команда:

```bash
cargo test --lib hidden_library
```

---

## Критерии готовности

- ✅ Все trained genomes автоматически сохраняются в hidden archive
- ✅ Entry содержит genealogy + objectives + profiler
- ✅ API list/unhide/delete работает
- ✅ Фильтрация в backend корректна
- ✅ Тесты проходят

---

## Вывод

- Изменения в backend persistence/API слоях
- Основа для T110 (weights export) и T111 (archive UI)
