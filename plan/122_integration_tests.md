# Task 122: Integration Tests

**Phase**: 4 (Testing)  
**Complexity**: High  
**Estimate**: 12h  
**Dependencies**: All Phase 2 + Task 117-119  
**Owner**: Backend + Frontend

---

## Goal

Validate end-to-end integration between backend commands and frontend FSD modules.

---

## Required Scenarios

1. Multi-objective run with Pareto updates by generation.
2. Device constraints run with MOPS/RAM/FLASH/latency feasibility.
3. Hidden autosave + hidden archive page workflow.
4. Genealogy pipeline from backend lineage to frontend tree visualization.
5. Stopping criteria trigger + reason propagation.
6. Parallel orchestration + profiler payload consistency.

---

## FSD Integration Points

- `src/pages/evolution-studio-page/*`
- `src/pages/hidden-archive-page/*`
- `src/features/evolution-manager/*`
- `src/features/evolution-studio/*`
- `src/features/genome-library/*`
- `src/widgets/evolution-dashboard/*`
- `src/widgets/pareto-front-visualizer/*`
- `src/widgets/genealogy-tree-viewer/*`

---

## Steps

### 1. Prepare fixtures

- dataset profile
- device templates
- backend command fixtures

### 2. Implement integration suites

Place tests in frontend and backend integration locations as appropriate.

### 3. Validate route-level behavior

Check `/evolution-studio`, `/genome-library`, `/hidden-archive` flows.

---

## Run

```bash
npx vitest run --config vitest.config.ts
cargo test --test integration_*
```

---

## Done Criteria

- [ ] All required scenarios pass
- [ ] No contract mismatch between backend and frontend
- [ ] No flaky behavior

---

## Output

- Integration test suites across frontend and backend
- Verified page-feature-widget-backend data flow
