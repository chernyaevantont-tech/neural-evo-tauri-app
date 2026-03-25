# Task 124: Soak and Stress Tests

**Phase**: 4 (Testing)  
**Complexity**: Medium  
**Estimate**: 4h setup + 72h run  
**Dependencies**: All tasks completed  
**Owner**: QA / Backend

---

## Goal

Validate long-run stability, memory behavior, and performance degradation risks.

---

## Scenarios

1. Long evolution soak: 24-48h continuous run.
2. Hidden archive growth under repeated runs.
3. Device constraints churn with frequent template switching.
4. Dashboard stress with frequent live-metric updates.

---

## Metrics

- crash-free runtime
- memory growth trend
- throughput drift
- UI latency drift
- archive query degradation

---

## FSD Focus

Observe degradation on real runtime compositions:
- `src/pages/evolution-studio-page/*`
- `src/pages/hidden-archive-page/*`
- `src/widgets/evolution-dashboard/*`
- `src/widgets/post-evolution-panel/*`

---

## Steps

### 1. Prepare run profile

Use fixed dataset and fixed device constraints (MOPS/RAM/FLASH/latency).

### 2. Execute 72h run

Run in background with periodic snapshots of logs and memory stats.

### 3. Produce report

Include memory curves, failure counts, worst-case latency, and conclusions.

---

## Done Criteria

- [ ] No critical crashes during soak window
- [ ] No unbounded memory growth
- [ ] Dashboard and archive remain usable
- [ ] Final soak report produced

---

## Output

- Soak scripts/config
- Final stability report
