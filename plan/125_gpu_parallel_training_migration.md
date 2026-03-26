# Task 125: Real NVIDIA GPU Parallel Training (Burn + Tauri)

## Goal
Enable true multi-model concurrent training on NVIDIA GPU with predictable throughput and stable telemetry.

## Current blocker
The app starts multiple evaluation workers in parallel, but they share one WGPU device/queue in-process. This produces parallel tasks but usually serialized GPU execution.

## Phase 0: Baseline and guardrails

1. Add explicit backend/mode runtime diagnostics
- Log backend kind (`wgpu`, `cuda`) and effective parallel workers for each run.
- Log queue wait time per genome.

2. Add throughput baseline script
- Run 20 genomes with workers K=1/2/3/4.
- Collect wall-clock and aggregate samples/sec.
- Keep this baseline before migration.

Acceptance:
- We can compare performance before/after by exact metrics.

## Phase 1: Backend abstraction (no behavior change yet)

1. Create backend module and aliases
- New file: `src-tauri/src/backend.rs`
- Define backend/device aliases in one place.

Example shape:
```rust
// backend.rs
use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;

pub type TrainBackend = Autodiff<Wgpu>;
pub type TrainDevice = WgpuDevice;

pub fn create_device() -> TrainDevice {
    WgpuDevice::default()
}

pub fn backend_name() -> &'static str {
    "wgpu"
}
```

2. Replace hardcoded aliases in runtime files
- `src-tauri/src/lib.rs`
- `src-tauri/src/data_loader.rs`
- `src-tauri/src/csv_loader.rs`

Acceptance:
- `cargo check` passes.
- Runtime behavior unchanged.

## Phase 2: Process worker architecture (key for real GPU concurrency)

1. Introduce worker binary mode
- Extend `src-tauri/src/main.rs` to support:
  - Normal Tauri app mode
  - Worker mode via CLI flag, e.g. `--train-worker`

2. Worker request/response contracts
- New DTOs in `src-tauri/src/dtos.rs`:
  - `WorkerTrainRequest`
  - `WorkerTrainProgress`
  - `WorkerTrainResult`

3. IPC transport
- Use stdio JSON-lines between parent and worker process.
- Parent sends request once, receives progress events and final result.

4. Parent orchestrator in app process
- For `parallel-safe-limited`, spawn up to K worker processes.
- Queue remaining genomes.
- Keep deterministic result ordering by index.

Acceptance:
- With K>1, at least two workers are active concurrently from parent viewpoint.
- Progress events continue to update UI per genome.

## Phase 3: CUDA backend path for workers

1. Add CUDA backend feature configuration in `Cargo.toml`
- Keep `wgpu` path available as fallback.
- Add compile-time feature gates for `cuda` worker mode.

2. In worker mode only, create CUDA device/backend
- Use `backend.rs` to switch worker backend to CUDA alias.
- Keep main Tauri process backend-agnostic.

3. Per-worker memory cap
- Respect worker-level VRAM estimate and reject launch if over cap.

Acceptance:
- Worker logs backend name `cuda`.
- Throughput improves from K=1 to K=2 in benchmark.

## Phase 4: Telemetry and ETA correctness

1. Emit worker-level timing
- queue_wait_ms
- gpu_active_ms
- step_time_ms EMA

2. UI ETA model
- Per-job ETA from worker EMA.
- Generation ETA from active workers + queued jobs.

3. Dashboard fields
- effective_gpu_workers
- queued_jobs
- avg_step_ms

Acceptance:
- ETA converges during run and no longer stays zero/unavailable.

## Phase 5: Safety and fallback

1. Automatic fallback policy
- On worker crash/OOM:
  - reduce K
  - retry once
  - optionally fallback to sequential

2. Cancellation correctness
- Stop command must terminate running worker processes and drain queues.

3. Soak tests
- 100+ genomes, no deadlocks, no zombie worker processes.

Acceptance:
- Stable long run with controlled failure behavior.

## Implementation order in this repo

1. `src-tauri/src/backend.rs` (new)
2. Refactor aliases in:
   - `src-tauri/src/lib.rs`
   - `src-tauri/src/data_loader.rs`
   - `src-tauri/src/csv_loader.rs`
3. Worker mode plumbing in:
   - `src-tauri/src/main.rs`
   - `src-tauri/src/lib.rs`
   - `src-tauri/src/dtos.rs`
4. Parent worker-pool scheduler in `src-tauri/src/lib.rs`
5. UI ETA updates in:
   - `src/features/evolution-studio/model/useEvolutionLoop.ts`
   - `src/widgets/evolution-dashboard/model/dashboardSelectors.ts`

## Practical note for Windows + WDDM
`nvidia-smi` per-process compute metrics are often limited under WDDM. Use wall-clock and aggregate samples/sec as source of truth for parallel speedup.

## Done criteria for Task 125
- K>1 yields measurable speedup vs K=1 on same dataset/population.
- UI shows non-zero, stable ETAs.
- Worker crashes do not kill full run.
- No regression in sequential mode.
