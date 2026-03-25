# Задача 107: Genealogy Tracking (Backend)

**Фаза**: 2 (Core Features - Genealogy)  
**Сложность**: Medium  
**Время**: 8 часов  
**Зависимости**: Task 001 (DTO contracts)  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Добавить полноценное отслеживание происхождения геномов в эволюции:
- parent-child связи
- тип мутации/операции (AddNode, RemoveNode, Crossover, ParameterMutation...)
- generation/time метаданные
- проверка отсутствия циклов в genealogy graph

Также добавить API для получения ancestry chain выбранного genome.

---

## Входные данные

- `src-tauri/src/dtos.rs` (genealogy DTO)
- `src-tauri/src/lib.rs` (Tauri commands)
- `src-tauri/src/entities.rs` (evolution runtime)
- `plan.md` раздел 21

---

## Пошаговое выполнение

### Шаг 1: Создать `genealogy.rs`

Создать:
- `src-tauri/src/genealogy.rs`

Базовые структуры:

```rust
pub struct GenomeLineageRecord {
    pub genome_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,
    pub mutation_type: MutationType,
    pub created_at_unix_ms: u64,
}

pub struct GenealogyGraph {
    pub nodes: HashMap<String, GenomeLineageRecord>,
    pub edges: Vec<(String, String)>, // parent -> child
}
```

---

### Шаг 2: Реализовать логирование lineage событий

Добавить функции:
- `register_founder(genome_id, generation)`
- `register_mutation(parent_id, child_id, mutation_type, generation)`
- `register_crossover(parent_a, parent_b, child_id, generation)`

Интегрировать вызовы в mutation pipeline.

---

### Шаг 3: Реализовать проверку циклов

Добавить DFS/Kahn проверку DAG:

```rust
pub fn validate_acyclic(graph: &GenealogyGraph) -> Result<(), GenealogyError>
```

Если цикл обнаружен:
- логировать ошибку
- не коммитить mutation record

---

### Шаг 4: Реализовать query API

Добавить:
- `get_genealogy(genome_id) -> GenealogyPath`
- `get_ancestors(genome_id, depth)`
- `get_descendants(genome_id, depth)`

---

### Шаг 5: Зарегистрировать Tauri commands

В `lib.rs`:
- `get_genealogy`
- `get_ancestors`
- `get_descendants`

---

## Тесты

- Unit tests:
  - founder без родителей
  - mutation с одним родителем
  - crossover с двумя родителями
  - корректность ancestry traversal
  - rejection циклической связи
- Integration:
  - эмуляция 3-4 поколений и проверка целостности графа

Команда:

```bash
cargo test --lib genealogy
```

---

## Критерии готовности

- ✅ В backend хранится lineage по каждому genome
- ✅ Поддержаны mutation и crossover события
- ✅ Граф генеалогии валидируется как acyclic
- ✅ API ancestry/descendants работает
- ✅ Тесты проходят

---

## Вывод

- Новый файл: `src-tauri/src/genealogy.rs`
- Изменения: `src-tauri/src/entities.rs`, `src-tauri/src/lib.rs`
- Используется в T108 (visualization) и T109 (hidden library metadata)
