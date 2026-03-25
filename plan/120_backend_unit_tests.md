# Task 120: Backend Unit Tests

**Phase**: 4 (Testing)  
**Complexity**: Medium  
**Estimate**: 10h  
**Dependencies**: Task 101-116  
**Owner**: Backend (Rust)

---

## Goal

Add unit tests for all backend modules that power the new evolution stack.

This is backend-only, but test assertions must validate DTO/API behavior expected by frontend FSD layers.

---

## Target Modules

- `src-tauri/src/profiler.rs`
- `src-tauri/src/pareto.rs`
- `src-tauri/src/device_profiles.rs`
- `src-tauri/src/device_library.rs`
- `src-tauri/src/genealogy.rs`
- `src-tauri/src/stopping_criteria.rs`
- `src-tauri/src/orchestrator/*`

---

## Steps

### 1. Build a test matrix

For each module include:
- happy path
- boundary cases
- invalid input
- serialization compatibility

### 2. Add inline unit tests

Use `#[cfg(test)]` in each target module.

### 3. Cover critical behaviors

- Pareto: dominance and frontier stability
- Device constraints: MOPS/RAM/FLASH/latency feasibility and penalty growth
- Device library: CRUD, uniqueness, import/export merge semantics
- Genealogy: DAG integrity and traversal
- Stopping: any/all policy and plateau logic

### 4. Validate contract compatibility

Ensure JSON payloads match frontend expectations.

---

## Run

```bash
cargo test --lib
```

---

## Done Criteria

- [ ] All core backend modules have unit tests
- [ ] Edge cases for resource constraints are covered
- [ ] No flaky tests
- [ ] CI pass

---

## Output

- Updated `src-tauri/src/*` test blocks
- Stable backend baseline for tasks 121-124
