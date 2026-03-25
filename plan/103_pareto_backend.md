# Задача 103: Pareto Front Computation (Backend)

**Фаза**: 2 (Core Features)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Реализовать backend вычисление Pareto front для multi-objective эволюции:
- dominance relation
- frontier computation O(N^2)
- payload generation для фронтенд-визуализации

---

## Входные данные

- `src-tauri/src/pareto.rs` (создать)
- `src-tauri/src/dtos.rs`
- `src-tauri/src/lib.rs`

---

## Пошаговое выполнение

### Шаг 1: Реализовать dominance relation

`is_dominated(a, b)` с учетом целей accuracy/latency/model_size.

### Шаг 2: Реализовать frontier computation

`compute_pareto_front(genomes)` с O(N^2).

### Шаг 3: Добавить Tauri command

Экспортировать `compute_pareto_front` в `lib.rs`.

### Шаг 4: Сформировать `GenerationParetoFront`

Вернуть структуру, пригодную для `widgets/pareto-front-visualizer`.

---

## Тесты

```bash
cargo test --lib pareto
```

Проверить:
- корректность frontier на известных примерах
- стабильность для равных objective vectors
- performance sanity для N=100+

---

## Критерии готовности

-  Pareto backend модуль реализован
-  Command доступна для фронтенда
-  DTO-контракт совместим с текущим `src/shared/lib/dtos.ts`
-  Тесты проходят

---

## Вывод

- Изменения: `src-tauri/src/pareto.rs`, `src-tauri/src/lib.rs`
- Основа для T104/T118/T119

