# Задача 115: Device Library Persistence (Backend)

**Фаза**: 2 (Core Features - Device Constraints UX)  
**Сложность**: Medium  
**Время**: 7 часов  
**Зависимости**: Task 001, Task 105  
**Выполнит**: Backend разработчик (Rust)

---

## Описание

Реализовать библиотеку пользовательских устройств, чтобы не вводить ограничения вручную каждый запуск.

Под устройством хранится именованный профиль ограничений:
- `mops_budget` (миллионы операций в секунду)
- `ram_budget_mb`
- `flash_budget_mb`
- `max_latency_ms`
- optional metadata (`notes`, `tags`, `created_at`, `updated_at`)

Библиотека должна поддерживать CRUD + импорт/экспорт.

---

## Входные данные

- `src-tauri/src/dtos.rs` (добавить DTO для device templates)
- `src-tauri/src/device_profiles.rs` (валидация constraints)
- `src-tauri/src/lib.rs` (Tauri commands)
- `plan.md` раздел 20 (расширение)

---

## Пошаговое выполнение

### Шаг 1: DTO и модель хранения

Добавить:

```rust
pub struct DeviceTemplateDto {
    pub id: String,
    pub name: String,
    pub constraints: DeviceResourceConstraints,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
}
```

---

### Шаг 2: Реализовать storage модуль

Создать:
- `src-tauri/src/device_library.rs`

Функции:
- `list_device_templates()`
- `create_device_template(input)`
- `update_device_template(id, patch)`
- `delete_device_template(id)`
- `duplicate_device_template(id, new_name)`

Хранение: JSON-файл в app data dir (или текущая persistence-стратегия проекта).

---

### Шаг 3: Валидация и уникальность

Правила:
- `name` обязателен, длина 2..64
- name unique (case-insensitive)
- budgets > 0
- `max_latency_ms > 0`

---

### Шаг 4: Import/Export библиотеки

Добавить:
- `export_device_library(path)`
- `import_device_library(path, mode: merge|replace)`

---

### Шаг 5: Tauri API

В `lib.rs` зарегистрировать:
- `list_device_templates`
- `create_device_template`
- `update_device_template`
- `delete_device_template`
- `duplicate_device_template`
- `export_device_library`
- `import_device_library`

---

## Тесты

- Unit:
  - CRUD операции
  - уникальность имен
  - валидация constraints
- Integration:
  - import/export roundtrip
  - merge/replace режимы

Команда:

```bash
cargo test --lib device_library
```

---

## Критерии готовности

- ✅ Есть персистентная библиотека пользовательских устройств
- ✅ Профили сохраняются между сессиями
- ✅ Поддержан полный CRUD
- ✅ Поддержан import/export
- ✅ API готово для frontend менеджера
- ✅ Тесты проходят

---

## Вывод

- Новый файл: `src-tauri/src/device_library.rs`
- Изменения: `src-tauri/src/lib.rs`, `src-tauri/src/dtos.rs`
- Закрывает требование пользователя «сохранить устройство и не вводить заново»
