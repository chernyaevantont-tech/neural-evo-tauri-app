# Задача 101: Performance Profiler Backend

**Фаза**: 2 (Core Features)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Инструментировать backend training pipeline для сбора profiler-метрик:
- train/val/test timing
- throughput
- peak active memory
- category peaks (интеграция с T114)

---

## Входные данные

- `src-tauri/src/entities.rs`
- `src-tauri/src/dtos.rs`
- `src-tauri/src/profiler.rs` (создать/расширить)

---

## Пошаговое выполнение

### Шаг 1: Реализовать collector

Создать `ProfilerCollector` с mark/start/end API для фаз обучения.

### Шаг 2: Интегрировать collector в training loop

В `run_eval_pass`, `run_validation_pass`, `run_test_pass` записывать времена и batch statistics.

### Шаг 3: Собирать throughput и peak memory

Обновлять `samples_per_sec`, `peak_active_memory_mb` и экспортировать в DTO.

### Шаг 4: Экспортировать profiler в training results

Включить `profiler` в payload итогов эволюции.

---

## Тесты

```bash
cargo test --lib profiler
```

Проверить:
- корректность таймингов
- монотонность обновления peak значений
- отсутствие падений на пустых/коротких прогонах

---

## Критерии готовности

-  Profiler данные приходят в результатах run
-  Метрики train/val/test заполнены
-  Подготовлена база для UI T102/T118/T119
-  Тесты проходят

---

## Вывод

- Изменения: `src-tauri/src/profiler.rs`, `src-tauri/src/entities.rs`
- Основа для performance-визуализации на фронтенде

