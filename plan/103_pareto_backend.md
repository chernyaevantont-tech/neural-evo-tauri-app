# Задача 103: Pareto Front Computation (Backend)

**Фаза**: 2 (Core Features - Multi-Objective)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001 (dtos)  
**Выполнит**: Backend разработчик (Rust)

---

## Краткое описание

Реализовать функции для вычисления Парето-фронта: dominance relation, frontier computation, O(N²) алгоритм. Интегрировать в `lib.rs` как новую Tauri command `compute_pareto_front`.

---

## Ключевые функции

```rust
pub fn is_dominated(a: &GenomeObjectives, b: &GenomeObjectives) -> bool {
    // A доминирует B если A лучше или равна B по всем целям,
    // и хотя бы по одной цели A строго лучше
    (b.accuracy >= a.accuracy) &&
    (b.inference_latency_ms <= a.inference_latency_ms) &&
    (b.model_size_mb <= a.model_size_mb) &&
    !(a.accuracy == b.accuracy && ...)
}

pub fn compute_pareto_front(genomes: &[GenomeObjectives]) -> Vec<GenomeObjectives> {
    // O(N²) frontier computation
}
```

---

## Этапы реализации

### 1. Создать модуль `src-tauri/src/pareto.rs`
### 2. Реализовать dominance relation и Парето-фронт вычисление
### 3. Добавить в lib.rs command `compute_pareto_front`
### 4. Написать unit и integration tests
### 5. Проверить компиляцию и performance (O(N²) < 100ms для N=100)

---

## Критерии готовности

- ✅ Модуль `pareto.rs` создан с dominance и frontier функциями
- ✅ Pareto front O(N²) < 100ms для N=100+
- ✅ Command `compute_pareto_front` доступна в Tauri
- ✅ Все тесты проходят
- ✅ Генерирует `GenerationParetoFront` DTO

---

## Вывод

- Файл: `src-tauri/src/pareto.rs` (~150 LOC)
- Зависимость: Task 104 (frontend visualization)

---

## Ссылки

- План.md раздел 19 (Multi-Objective Optimization)
- DTO: `GenomeObjectives`, `GenerationParetoFront`

