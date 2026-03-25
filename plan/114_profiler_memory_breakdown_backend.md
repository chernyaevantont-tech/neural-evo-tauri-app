# Задача 114: Profiler Memory Breakdown (Backend)

**Фаза**: 2 (Core Features - Metrics Quality)  
**Сложность**: Medium  
**Время**: 6 часов  
**Зависимости**: Task 101, Task 002  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Довести профайлер до полноценного memory breakdown без постоянных заглушек:
- `peak_model_params_mb`
- `peak_gradient_mb`
- `peak_optim_state_mb`
- `peak_activation_mb`

Добавить deterministic estimator + runtime/hybrid режимы сбора.

---

## Входные данные

- `src-tauri/src/profiler.rs`
- `src-tauri/src/entities.rs`
- `src-tauri/src/orchestrator/memory_estimator.rs` (если уже есть)
- `plan.md` раздел 18

---

## Пошаговое выполнение

### Шаг 1: Расширить profiler collector

Добавить методы:
- `set_model_params_mb(value)`
- `set_gradients_mb(value)`
- `set_optimizer_state_mb(value)`
- `set_activation_mb(value)`
- `update_peak_category(...)`

---

### Шаг 2: Реализовать оценку памяти по тензорам

Добавить helper-функции:
- `estimate_model_params_mb(model)`
- `estimate_gradients_mb(model)`
- `estimate_optimizer_state_mb(optimizer)`
- `estimate_activations_mb(batch_shapes)`

Формула: `bytes = elements * dtype_size`.

---

### Шаг 3: Поддержать режимы `estimate/runtime/hybrid`

Добавить конфиг:
- `profiling.memory_mode = "estimate" | "runtime" | "hybrid"`

Поведение:
- estimate: только расчет по формулам
- runtime: только telemetry
- hybrid: runtime, с fallback на estimate

---

### Шаг 4: Интеграция в train/val/test проходы

На каждом батче обновлять peaks по категориям.

---

### Шаг 5: Валидация payload

Проверить, что для нетривиальных сетей значения категорий не нулевые.

---

## Тесты

- Unit:
  - формулы memory estimate
  - monotonic peak updates
  - fallback в hybrid режиме
- Integration smoke:
  - Dense-only модель
  - `peak_*_mb > 0` для основных категорий

Команда:

```bash
cargo test --lib profiler
```

---

## Критерии готовности

- ✅ Убраны постоянные заглушки по memory breakdown
- ✅ Режимы estimate/runtime/hybrid работают
- ✅ Пики корректно агрегируются по категориям
- ✅ Payload стабилен и пригоден для UI
- ✅ Тесты проходят

---

## Вывод

- Изменения: `src-tauri/src/profiler.rs`, `src-tauri/src/entities.rs`
- Улучшает качество метрик для T102/T118/T119 и тестов T120/T122
