# Задача 117-119: Phase 3 Tasks - UI/UX Integration

**Фаза**: 3 (UI/UX Integration)  
**Зависимости**: All Phase 2 tasks

---

## Задача 117: Settings & Configuration UI Expansion

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Tasks 104-116

### Описание
Расширить UI для конфигурации всех новых features:

#### Sections добавить:
1. **Performance & Objectives**
   - Mode selector: Single-Objective (legacy) vs Multi-Objective (Pareto)
   - Secondary objectives checkboxes (Latency, Model Size, Training Time)
   - Emphasis sliders (optional)

2. **Device Targeting**
   - Dropdown: "Target Device" (9 built-in + Custom)
   - If Custom: fields для RAM, VRAM, latency_budget, max_model_size
   - Show: "Estimated parallelism for {device}: N"

3. **Stopping Criteria**
   - Add criterion button
   - List of active criteria with removable X
   - Per criterion: type dropdown, parameter inputs
   - Policy selector: "any" / "all"

4. **Advanced Performance** (collapsed)
   - Safety margin (MB)
   - Estimator safety factor
   - Profiling enabled checkbox

### Вывод
- Updated `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`
- 4 new section components
- Zustand store extensions (Task 003 covers types)

---

## Задача 118: Evolution Progress Dashboard

**Сложность**: High | **Время**: 10ч | **Зависимости**: Tasks 101-116

### Описание
Объединённая dashboard во время эволюции, отображающая все метрики:

#### Top Section: Overview Cards
- Generations Elapsed: X / Y
- Genomes Evaluated: X / total
- Current Best Fitness: 0.XX
- Pareto Front Size: N genomes
- Elapsed Time: HH:MM:SS
- ETA (if plateau detection available)

#### Middle Section: Charts
- Line plot: Best Fitness over generations
- Area plot: Population average fitness
- Shaded region: device constraints (if active)

#### Right Panel: Active Jobs (if parallelism > 1)
- Table: Job ID, Genome ID, Progress %, Status, Duration, ETA
- Color code: running (blue), completed (green), failed (red)

#### Bottom Tabs (4 tabs):
1. **Pareto Front**: 2D scatter (Accuracy vs Latency), size=model_size, highlighted non-dominated
2. **Genealogy Tree**: Interactive tree, nodes=genomes, edges=parent-child, labels=mutation types
3. **Stopping Criteria Progress**: Progress bars per criterion, plateau patience meter, time/gen limits
4. **Performance Metrics**: Per genome table (training time, inference latency, peak VRAM), charts

#### Control Buttons
- "Pause Evolution"
- "Resume Evolution"
- "Stop Evolution"
- "Save Checkpoint"

### Вывод
- Новый компонент `src/widgets/evolution-dashboard/EvolutionDashboard.tsx` (расширение)
- 5 tab components
- Charts с Chart.js / Recharts

---

## Задача 119: Post-Evolution Analysis Panel

**Сложность**: Medium | **Время**: 6ч | **Зависимости**: Tasks 104-116

### Описание
Панель после завершения эволюции для анализа и export:

#### Pareto Front Visualization
- 2D/3D scatter с selection
- Legend: feasible (meets device constraints), infeasible

#### Detailed Genome Comparison
- Multi-select 2-3 genomes
- Side-by-side: architecture, accuracy/latency/size, training/inference time, memory, genealogy, device compatibility

#### Genealogy Analysis
- Family tree visualization
- Timeline: generation → fitness progression
- Hover: show mutations
- Click: export ancestry JSON/GraphML

#### Hidden Library Management
- "N genomes auto-saved to hidden archive"
- Button: "View Archive"
- Quick stats: avg fitness, accuracy range

#### Export/Action Buttons
- "Download Pareto Front (JSON)"
- "Select & Export Model Weights"
- "Save Evolution Report (PDF)"
- "Continue Evolution"

### Вывод
- Компонент `src/widgets/post-evolution-panel/PostEvolutionPanel.tsx`
- Export functionality (JSON, PDF, weights)
- Archive preview

