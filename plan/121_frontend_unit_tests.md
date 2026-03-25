# Task 121: Frontend Unit Tests

**Phase**: 4 (Testing)  
**Complexity**: Medium  
**Estimate**: 8h  
**Dependencies**: Task 102, Task 104-119  
**Owner**: Frontend

---

## Goal

Add unit/component tests for the current FSD frontend structure.

Scope must cover:
- `features` (domain UI + state actions)
- `widgets` (composite visual modules)
- `pages` (composition correctness)
- `shared` (hooks and utility helpers)

---

## Target Areas

- `src/features/evolution-manager/ui/*`
- `src/features/evolution-studio/ui/*`
- `src/widgets/pareto-front-visualizer/*`
- `src/widgets/genealogy-tree-viewer/*`
- `src/widgets/evolution-dashboard/*`
- `src/widgets/post-evolution-panel/*`
- `src/pages/evolution-studio-page/*`
- `src/pages/hidden-archive-page/*`
- `src/shared/hooks/*`

---

## Steps

### 1. Prioritize critical components

- `DeviceProfileSelector`
- `DeviceLibraryManager`
- `StoppingCriteriaPanel`
- `ParetoScatterPlot`
- `EvolutionDashboard`
- `PostEvolutionPanel`

### 2. Cover store-driven behavior

Validate interactions with `useEvolutionSettingsStore`:
- apply device template
- feasible/infeasible filtering
- stopping policy changes
- objective mode toggles

### 3. Validate page composition

Ensure `EvolutionStudioPage` and `EvolutionSettingsPanel` integrate feature/widget modules correctly.

---

## Run

```bash
npx vitest run src/features src/widgets src/pages src/shared/hooks
```

---

## Done Criteria

- [ ] Critical frontend modules covered
- [ ] Tests reference current FSD paths only
- [ ] No flaky behavior
- [ ] One-shot run passes

---

## Output

- New and updated `*.test.ts(x)` files in FSD layers
- Stable frontend baseline for tasks 122-123
