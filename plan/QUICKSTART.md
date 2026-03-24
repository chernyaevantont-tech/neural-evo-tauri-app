# 🚀 QUICKSTART: Как начать работу с задачами

**Создано**: 2026-03-24

---

## 📋 Краткая справка

Проект полностью разбит на **124 независимых задач**, организованных в 4 фазы:

| Фаза | Задачи | Время | Описание |
|------|--------|-------|---------|
| 1️⃣ **Infrastructure** | T001-T003 (3) | 2 неделе | Foundation: DTOs, Orchestrator, Frontend types |
| 2️⃣ **Features** | T101-T113 (13) | 6 недель | Core features: Profiler, Pareto, Device, Genealogy, Archive, Stopping |
| 3️⃣ **UI Integration** | T117-T119 (3) | 2 недели | Dashboard, settings, post-evolution panels |
| 4️⃣ **Testing** | T120-T124 (5) | 2 неделе (+ 72h soak) | Unit, integration, E2E, soak tests |

**Total**: ~10-11 weeks (sequential) or 6-7 weeks (parallel with N agents)

---

## 📁 Структура задач

```
plan/
├── 00_INDEX.md                          ← START HERE (matrix, dependencies, execution modes)
├── 001_backend_dto_contracts.md         (Phase 1)
├── 002_orchestrator_module_scaffold.md  (Phase 1)
├── 003_frontend_store_extensions.md     (Phase 1)
├── 101_performance_profiler_backend.md  (Phase 2)
├── 102_performance_profiler_frontend.md (Phase 2)
├── 103_pareto_backend.md                (Phase 2)
├── 104-116_phase2_templates.md          (Phase 2: templates for T104-T113)
├── 117-119_phase3_ui_templates.md       (Phase 3: templates for T117-T119)
└── 120-124_phase4_testing_templates.md  (Phase 4: templates for T120-T124)
```

---

## ✅ Как выполнить задачу

### Step 1: Выберите задачу
Откройте матрицу в [00_INDEX.md](./00_INDEX.md), найдите номер задачи (e.g., **T101**)

### Step 2: Прочитайте задачу
Откройте соответствющий файл (e.g., `101_performance_profiler_backend.md`)

### Step 3: Проверьте зависимости
В секции "Зависимости" убедитесь, что все prerequisite tasks завершены

### Step 4: Выполните пошагово
1. Прочитайте "## Описание" - что делать
2. Прочитайте "## Входные данные" - какие файлы/документы нужны
3. Следуйте "## Пошаговое выполнение" - step-by-step инструкции
4. Запустите "## Тесты" - убедитесь, что всё работает
5. Проверьте "## Критерии готовности" (все ✅ должны быть отмечены)

### Step 5: Завершение
- Коммитьте код в репозиторий
- Отметьте задачу как "Done" в tracking системе
- Информируйте следующего агента о completion

---

## 🎯 Quick Reference: Какую задачу выбрать?

### Для Backend разработчика (Rust)
- **T001**: DTO contracts (foundation)
- **T002**: Orchestrator module
- **T101**: Performance profiler
- **T103**: Pareto computation
- **T105**: Device profiles
- **T107**: Genealogy tracking
- **T109**: Hidden library
- **T110**: Weight export
- **T112**: Stopping criteria
- **T120**: Unit tests (backend)
- **T122**: Integration tests

### Для Frontend разработчика (React/TypeScript)
- **T003**: Frontend store types & hooks
- **T102**: Profiler UI
- **T104**: Pareto visualization
- **T106**: Device selector UI
- **T108**: Genealogy tree viewer
- **T111**: Hidden archive page
- **T113**: Stopping criteria UI/panels
- **T117**: Settings expansion
- **T118**: Evolution dashboard
- **T119**: Post-evolution analysis
- **T121**: Unit tests (frontend)
- **T123**: E2E tests

### Для QA/Testing
- **T120**: Backend unit tests
- **T121**: Frontend unit tests
- **T122**: Integration tests
- **T123**: E2E tests
- **T124**: Soak & stress tests

---

## ⏱️ Параллельное выполнение (Multi-Agent Mode)

Если работает N агентов одновременно:

```
Week 1-2:  T001 || T002 || T003 (foundation layer)
           ↓ (все задачи Phase 1 должны быть done перед Phase 2)
           
Week 2-4:  T101, T102 || T103, T104 (2 агента параллельно)
Week 4-7:  T105, T106 || T107, T108 || T109, T110, T111 (3 агента)
Week 7-8:  T112, T113 (1 агент)
           ↓ (все Phase 2 должны быть done перед Phase 3)
           
Week 8-9:  T117 || T118 || T119 (3 агента UI параллельно)
           ↓
           
Week 9-10: T120 || T121 || T122 || T123 || T124 (5 агентов testing)
           T124 runs 72 hours in background
```

---

## 📊 Dependency Matrix (Simplified)

```
Phase 1 (Foundation)
└─ T001: DTOs
├─ T002: Orchestrator  
└─ T003: Frontend types

Phase 2 (Features)
├─ T101 ← T001; T102 ← T101
├─ T103 ← T001; T104 ← T103
├─ T105 ← T001; T106 ← T105, T104
├─ T107 ← T001; T108 ← T107
├─ T109 ← T001, T107, T101; T110 ← T109; T111 ← T110
└─ T112 ← T001; T113 ← T112

Phase 3 (UI)
├─ T117 ← T105, T112
├─ T118 ← T101-113 (all Phase 2)
└─ T119 ← T104-111

Phase 4 (Testing)
├─ T120 ← T101-113
├─ T121 ← T102, T104-113
├─ T122 ← all Phase 2
├─ T123 ← all Phases 1-3
└─ T124 ← all (long running)
```

---

## 💡 Pro Tips

### Tip 1: Используйте Template Tasks для экономии времени
Для T104-T113, основной шаблон уже подготовлен в `104-116_phase2_templates.md`. Просто расширьте и адаптируйте.

### Tip 2: Тесты должны быть встроены в каждую задачу
Каждый файл задачи содержит "## Тесты" секцию с конкретными тест-кейсами. Запустите их ДО завершения!

### Tip 3: Компиляция & Lint before committing
```bash
# Backend
cargo build --release
cargo test --lib

# Frontend
npm run build
npm run lint
npm run test
```

### Tip 4: Code Review Checklist
- [ ] Компилируется без ошибок и warnings
- [ ] Все тесты проходят
- [ ] Нет неиспользованных импортов
- [ ] Следует naming conventions проекта
- [ ] Документация (comments) добавлена для сложных функций
- [ ] Zero security issues

---

## 🔍 Содержимое каждого файла задачи

Каждый файл содержит стандартные секции:

```markdown
# Задача NNN: Название

**Фаза**: X | **Сложность**: Low/Medium/High | **Время**: Xч | **Зависимости**: T00X, T00Y

---

## Описание
Что нужно сделать (1-2 параграфа)

## Входные данные
- Какие файлы/документы нужны
- Ссылки на план.md
- DTO/типы из Task 001 и т.д.

## Пошаговое выполнение
### Шаг 1: ...
### Шаг 2: ...
(Code snippets + explanations)

## Критерии готовности
- ✅ Пункт 1
- ✅ Пункт 2
- ...

## Тесты
(Unit test examples + run commands)

## Вывод
- Файлы: ...
- LOC: ...
- Зависимость: ...
```

---

## 🚨 Если вы застряли

1. **Прочитайте интефейс TODO**
   - Каждая задача содержит "## Входные данные" - проверьте, всё ли там

2. **Проверьте зависимости**
   - Убедитесь, что все prerequisite tasks done
   - Смотрите матрицу в 00_INDEX.md

3. **Смотрите соответствующий раздел План.md**
   - Каждая задача ссылается на План.md chapter (e.g., "раздел 18")
   - Там есть полный контекст и design details

4. **Используйте Explore agent**
   - Если нужно понять текущую кодовую базу, запропите `Explore` агента
   - "Analyze current [component] implementation and explain structure"

---

## 📈 Progress Tracking

| Phase | Tasks | Status |
|-------|-------|--------|
| 1️⃣ Infrastructure | T001-T003 | ✅ Ready to execute |
| 2️⃣ Features | T101-T113 | ✅ Ready to execute |
| 3️⃣ UI Integration | T117-T119 | ✅ Ready to execute |
| 4️⃣ Testing | T120-T124 | ✅ Ready to execute |

---

## 📚 Дополнительные ресурсы

1. **plan.md** (1800+ lines)
   - Детальное описание архитектуры, API, UI, тесты
   - ДО выполнения задач прочитайте соответствующий раздел

2. **copilot-instructions.md**
   - Общие инструкции для проекта
   - Architecture overiew, tech stack, patterns

3. **Текущий код**
   - Для понимания существующей структуры
   - Используйте "Explore" agent для быстрого анализа

---

## ✨ Финальный бонус: Estimation

- **Total LOC Added**: ~3500 (backend 1000, frontend 1800, tests 700)
- **Total Test Cases**: 80+ (unit + integration + E2E)
- **Estimated Effort**: 10-11 weeks (1 dev)  OR  6-7 weeks (N agents parallel)
- **Success Rate Target**: 100% of features working, >80% test coverage

---

**Last Updated**: 2026-03-24  
**Version**: 1.0 (Initial Release)  
**Status**: 🟢 Ready for execution

Start with [00_INDEX.md](./00_INDEX.md) and pick your first task! 🚀
