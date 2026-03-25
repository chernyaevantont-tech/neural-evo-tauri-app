# Задача 105: Device Constraints Engine (Backend)

**Фаза**: 2 (Core Features - Device Constraints)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001 (DTO), Task 103 (Pareto backend)  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Реализовать backend-движок ограничений целевого устройства. Под "целевым устройством" понимается набор лимитов ресурсов, задаваемых пользователем:

- `mops_budget` (миллионы операций в секунду)
- `ram_budget_mb` (оперативная память)
- `flash_budget_mb` (постоянная память/flash)
- `max_latency_ms` (допустимая задержка инференса)

Также добавить 9 встроенных профилей для быстрого старта и API для расчета feasibility + penalty score.

---

## Входные данные

- `src-tauri/src/dtos.rs` (типы устройства и objectives)
- `src-tauri/src/pareto.rs` (оценка candidates)
- `src-tauri/src/lib.rs` (регистрация Tauri commands)
- `plan.md` раздел 20

---

## Пошаговое выполнение

### Шаг 1: Создать модуль `device_profiles.rs`

Создать:
- `src-tauri/src/device_profiles.rs`

Базовые типы:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceResourceConstraints {
    pub mops_budget: f32,
    pub ram_budget_mb: f32,
    pub flash_budget_mb: f32,
    pub max_latency_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceValidationResult {
    pub is_feasible: bool,
    pub violation_score: f32,
    pub mops_ratio: f32,
    pub ram_ratio: f32,
    pub flash_ratio: f32,
    pub latency_ratio: f32,
}
```

---

### Шаг 2: Добавить встроенные профили

Реализовать:

```rust
pub fn built_in_profiles() -> Vec<DeviceProfileDto> { /* 9 профилей */ }
```

Минимальный набор:
- Embedded MCU
- Edge Tiny
- Mobile Low-End
- Mobile Mid-Range
- Laptop CPU
- Laptop iGPU
- Workstation
- Cloud T4
- Cloud A100

Для каждого профиля определить реалистичные лимиты по MOPS/RAM/FLASH/latency.

---

### Шаг 3: Реализовать расчет feasibility и штрафа

```rust
pub fn validate_genome_for_device(
    objectives: &GenomeObjectives,
    constraints: &DeviceResourceConstraints,
) -> DeviceValidationResult
```

Формула штрафа:
- Для каждого превышенного ограничения считать `excess = max(0, ratio - 1.0)`
- Общий штраф: `penalty = sum(excess^2 * weight_i)`

Рекомендованные веса:
- latency = 0.35
- mops = 0.30
- ram = 0.20
- flash = 0.15

---

### Шаг 4: Интегрировать в fitness pipeline

Добавить helper:

```rust
pub fn apply_device_penalty(base_fitness: f32, violation_score: f32, alpha: f32) -> f32 {
    base_fitness - alpha * violation_score
}
```

Использовать во время ранжирования кандидатов (после подсчета objectives).

---

### Шаг 5: Открыть Tauri API

В `lib.rs` зарегистрировать команды:
- `get_device_profiles()`
- `validate_genome_for_device(genome_objectives, constraints)`
- `apply_device_penalty(base_fitness, violation_score, alpha)`

---

## Тесты

- Unit tests в `device_profiles.rs`:
  - profile list содержит 9 профилей
  - feasible genome получает `is_feasible = true`, `violation_score = 0`
  - для превышения RAM/FLASH/MOPS latency штраф > 0
  - штраф возрастает квадратично при росте превышения
- Integration:
  - через Tauri command проверить сериализацию/десериализацию

Команда:

```bash
cargo test --lib device_profiles
```

---

## Критерии готовности

- ✅ Реализован backend engine для MOPS/RAM/FLASH/latency constraints
- ✅ Добавлены 9 встроенных профилей
- ✅ Работает расчет feasibility и violation score
- ✅ Добавлена интеграция штрафа в fitness pipeline
- ✅ Tauri API доступен для фронтенда
- ✅ Тесты проходят

---

## Вывод

- Новый файл: `src-tauri/src/device_profiles.rs`
- Изменения: `src-tauri/src/lib.rs`, возможно `src-tauri/src/entities.rs`
- База для T106 (UI selector) и T115/T116 (библиотека пользовательских устройств)
