# Задачи 104-116: Индекс подробных файлов

Ранее здесь был общий шаблон. Задачи разнесены по отдельным подробным документам.

## Список задач

- [Задача 104: Pareto Front Visualization (Frontend)](./104_pareto_front_visualization_frontend.md)
- [Задача 105: Device Constraints Engine (Backend)](./105_device_profiles_backend.md)
- [Задача 106: Device Constraints UI (Frontend)](./106_device_profiles_frontend.md)
- [Задача 107: Genealogy Tracking (Backend)](./107_genealogy_tracking_backend.md)
- [Задача 108: Genealogy Tree Visualization (Frontend)](./108_genealogy_tree_visualization_frontend.md)
- [Задача 109: Hidden Library (Backend)](./109_hidden_library_backend.md)
- [Задача 110: Weight Checkpointing & Export](./110_weight_checkpointing_export.md)
- [Задача 111: Hidden Archive UI](./111_hidden_archive_ui.md)
- [Задача 112: Stopping Criteria (Backend)](./112_stopping_criteria_backend.md)
- [Задача 113: Stopping Criteria (Frontend)](./113_stopping_criteria_frontend.md)
- [Задача 114: Profiler Memory Breakdown (Backend)](./114_profiler_memory_breakdown_backend.md)
- [Задача 115: Device Library Persistence (Backend)](./115_device_library_backend.md)
- [Задача 116: Device Library Manager UI (Frontend)](./116_device_library_frontend.md)

## Комментарий по целевому устройству

В задачах 105/106/115/116 целевое устройство трактуется как набор ресурсных ограничений, задаваемых пользователем:

- MOPS (миллионы операций в секунду)
- RAM (MB)
- FLASH (MB)
- Max latency (ms)

Дополнительно реализуется библиотека пользовательских устройств с сохранением между сессиями (чтобы не вводить параметры повторно).

