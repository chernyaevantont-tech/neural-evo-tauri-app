# Task 123: E2E Tests (Tauri + UI)

**Phase**: 4 (Testing)  
**Complexity**: High  
**Estimate**: 10h  
**Dependencies**: All Phase 1-3  
**Owner**: QA / Frontend

---

## Goal

Cover real user workflows in Tauri UI from configuration to export.

---

## Required Workflows

1. Full evolution flow: settings -> run -> dashboard -> post-analysis.
2. Device template flow: save/load templates from device library.
3. Constraint flow: feasible/infeasible behavior on Pareto view.
4. Hidden archive flow: autosave -> archive -> unhide -> reseed.
5. Export flow: model weights + metadata.

---

## FSD Coverage Targets

Pages:
- `src/pages/evolution-studio-page/EvolutionStudioPage.tsx`
- `src/pages/genome-library-page/GenomeLibraryPage.tsx`
- `src/pages/hidden-archive-page/HiddenArchivePage.tsx`

Widgets:
- `src/widgets/evolution-dashboard/*`
- `src/widgets/post-evolution-panel/*`

---

## Steps

### 1. Prepare harness

Use Playwright or Tauri WebDriver.

### 2. Implement flows

Store scenarios in `e2e_tests/*.e2e.ts`.

### 3. Stabilize tests

Use explicit waits for async state transitions and avoid brittle selectors.

---

## Run

```bash
npm run test:e2e
```

---

## Done Criteria

- [ ] All workflows pass
- [ ] Tests cover current FSD pages/widgets
- [ ] No flaky behavior on repeated runs

---

## Output

- E2E suite for critical user journeys
- Verified runtime UX and export flows
