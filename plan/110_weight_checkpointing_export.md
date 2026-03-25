# Задача 110: Weight Checkpointing & Export

**Фаза**: 2 (Core Features - Persistence)  
**Сложность**: High  
**Время**: 8 часов  
**Зависимости**: Task 101, Task 109  
**Выполнит**: Backend + Frontend разработчик

---

## Описание

Добавить экспорт весов trained моделей для выбранного genome:
- сохранение в `.safetensors`
- metadata рядом в `metadata.json`
- повторное использование cached weights (если уже были сохранены)

---

## Входные данные

- `src-tauri/src/entities.rs` (model build/train)
- `src-tauri/src/lib.rs` (commands)
- hidden library из T109
- `plan.md` раздел 22

---

## Пошаговое выполнение

### Шаг 1: Реализовать backend weight saver

Создать модуль:
- `src-tauri/src/weight_io.rs`

Функции:
- `save_weights(genome_id, model, output_dir)`
- `load_weights(genome_id, input_dir)`
- `export_with_metadata(genome_id, output_dir, objectives, profiler)`

---

### Шаг 2: Добавить формат metadata.json

Минимальные поля:
- `genome_id`
- `created_at`
- `accuracy`
- `inference_latency_ms`
- `model_size_mb`
- `train_duration_ms`
- `device_profile_id` (если применимо)
- `lineage` (parent ids)

---

### Шаг 3: Tauri API

Добавить команды:
- `export_genome_with_weights(genome_id, output_path)`
- `has_cached_weights(genome_id)`

---

### Шаг 4: Frontend export flow

Добавить диалог:
- выбор genome
- выбор output path
- прогресс и статус
- отображение ошибок

---

### Шаг 5: Интеграция с hidden library

Если genome в hidden archive:
- использовать сохраненные метаданные
- не пересчитывать objectives при экспорте

---

## Тесты

- Unit backend:
  - save/load весов
  - metadata schema validation
- Integration:
  - export command создает `.safetensors` + `metadata.json`
- Frontend:
  - happy-path export flow
  - error state при invalid path

Команды:

```bash
cargo test --lib weight_io
npx vitest run src/features/genome-library
```

---

## Критерии готовности

- ✅ Весы экспортируются в `.safetensors`
- ✅ Metadata сохраняется рядом
- ✅ API для export/cached-status работает
- ✅ Frontend flow завершает экспорт без ручных шагов вне UI
- ✅ Тесты проходят

---

## Вывод

- Новый backend модуль для весов
- Расширение hidden library сценариев
- Используется в T111/T119
