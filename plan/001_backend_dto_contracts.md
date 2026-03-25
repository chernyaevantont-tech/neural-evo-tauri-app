# Задача 001: Backend DTO Contracts

**Фаза**: 1 (Infrastructure)  
**Сложность**: Medium  
**Время**: 4 часа  
**Зависимости**: None  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Определить и стабилизировать DTO-контракты между backend и frontend для:
- profiling
- pareto objectives
- device constraints и device library
- genealogy
- stopping criteria
- hidden archive metadata

FSD-примечание: задача backend-only, но контракты должны быть совместимы с текущей структурой фронтенда (`pages -> features -> widgets -> shared`).

---

## Входные данные

- `src-tauri/src/dtos.rs`
- `src/shared/lib/dtos.ts` (целевой mirror-контракт)
- `src/features/evolution-manager/model/store.ts`

---

## Пошаговое выполнение

### Шаг 1: Сгруппировать DTO по доменам

В `dtos.rs` выделить секции:
- Profiling
- Objectives/Pareto
- Device Constraints + Device Templates
- Genealogy
- Stopping
- Hidden Library Entry

### Шаг 2: Зафиксировать типы полей и единицы измерения

Особенно для resource constraints:
- MOPS
- RAM MB
- FLASH MB
- latency ms

### Шаг 3: Обновить payload структуры training/evolution ответов

Добавить опциональные поля, чтобы старые флоу не ломались при частичном внедрении.

### Шаг 4: Проверить serde-совместимость

Сериализация/десериализация для всех новых структур должна быть детерминированной.

---

## Тесты

```bash
cargo test --lib
```

Проверить:
- roundtrip serialize/deserialize
- default значения для optional полей
- совместимость с frontend типами

---

## Критерии готовности

-  DTO покрывают все Phase 2-4 домены
-  Поля ресурсов устройства заданы в MOPS/RAM/FLASH/latency
-  Контракты совместимы с текущим `src/shared/lib/dtos.ts`
-  Тесты проходят

---

## Вывод

- Изменения: `src-tauri/src/dtos.rs`
- Основа для всех следующих задач

