# Задача 003: Frontend Store Extensions

**Фаза**: 1 (Infrastructure)  
**Сложность**: Low  
**Время**: 4 часа  
**Зависимости**: Task 001  
**Выполнит**: Frontend разработчик

---

## Описание

Расширить Zustand stores в `src/entities/canvas-genomae/model/store.ts` и `src/features/evolution-manager/model/store.ts` новыми полями для tracking profiler data, Pareto front, genealogy, device profiles и stopping criteria.

---

## Входные данные

- Существующие stores (canvas-genome store, evolution-manager store)
- DTO из Task 001 (TypeScript версия)
- План.md раздел 24 (UI состояние modelling)

---

## Пошаговое выполнение

### Шаг 1: Создать TypeScript versions новых DTOs

Создать файл `src/shared/lib/dtos.ts`:

```typescript
// Profiling
export interface TrainingProfiler {
  train_start_ms: number;
  first_batch_ms: number;
  train_end_ms: number;
  total_train_duration_ms: number;
  val_start_ms: number;
  val_end_ms: number;
  val_duration_ms: number;
  test_start_ms: number;
  test_end_ms: number;
  test_duration_ms: number;
  peak_active_memory_mb: number;
  peak_model_params_mb: number;
  peak_gradient_mb: number;
  peak_optim_state_mb: number;
  peak_activation_mb: number;
  samples_per_sec: number;
  inference_msec_per_sample: number;
  batch_count: number;
  early_stop_epoch?: number;
}

// Pareto
export interface GenomeObjectives {
  genome_id: string;
  accuracy: number;
  inference_latency_ms: number;
  model_size_mb: number;
  training_time_ms: number;
  is_dominated: boolean;
  domination_count: number;
}

export interface GenerationParetoFront {
  generation: number;
  total_genomes: number;
  pareto_members: GenomeObjectives[];
  objectives_3d: [number, number, number][];
}

// Device Profile
export type ComputeType = "ARM" | "X86" | "GPU";

export interface DeviceProfile {
  device_id: string;
  device_name: string;
  compute_capability: ComputeType;
  ram_mb: number;
  vram_mb?: number;
  inference_latency_budget_ms: number;
  training_available: boolean;
  power_budget_mw?: number;
  max_model_size_mb?: number;
  target_fps?: number;
}

// Genealogy
export type MutationType =
  | { type: "Random" }
  | { type: "AddNode"; node_type: string; source: string; target: string }
  | { type: "RemoveNode"; node_id: string }
  | { type: "RemoveSubgraph"; node_ids: string[] }
  | { type: "ParameterMutation"; layer_id: string; param_name: string }
  | { type: "ParameterScale"; layer_id: string; scale_factor: number }
  | { type: "Crossover"; parent1: string; parent2: string };

export interface GenomeGeneology {
  genome_id: string;
  generation: number;
  parent_ids: string[];
  mutation_type: MutationType;
  mutation_params: Record<string, any>;
  fitness: number;
  accuracy: number;
  created_at_ms: number;
}

// Stopping Criteria
export type StoppingCriterionType =
  | { type: "GenerationLimit"; max_generations: number }
  | { type: "FitnessPlateau"; patience_generations: number; improvement_threshold: number; monitor: "best_fitness" | "pareto_coverage" | "population_avg" }
  | { type: "TimeLimit"; max_seconds: number }
  | { type: "TargetAccuracy"; threshold: number }
  | { type: "ManualStop" };

export interface StoppingPolicy {
  criteria: StoppingCriterionType[];
  policy_type: "any" | "all";
}
```

### Шаг 2: Расширить PopulatedGenome (canvas-genome уровень)

В `src/entities/canvas-genome/index.ts` или в типе, найти `PopulatedGenome` interface и добавить:

```typescript
export interface PopulatedGenome {
  // ... existing fields ...
  
  // NEW: Performance
  profiler?: TrainingProfiler;
  
  // NEW: Multi-objective
  objectives?: GenomeObjectives;
  is_dominated?: boolean;
  
  // NEW: Genealogy
  generation?: number;
  parent_ids?: string[];
  mutation_type?: MutationType;
  mutation_params?: Record<string, any>;
}
```

### Шаг 3: Расширить GenerationSnapshot

Найти `GenerationSnapshot` interface в `src/features/evolution-studio/model/useEvolutionLoop.ts` или в store:

```typescript
export interface GenerationSnapshot {
  // ... existing fields ...
  generation: number;
  genomes: PopulatedGenome[];
  bestFitness: number;
  avgNodes: number;
  timestamp: string;
  evaluated: boolean;
  
  // NEW: Genealogy
  genealogy?: Map<string, GenomeGeneology>;
  
  // NEW: Pareto front
  paretoFront?: GenomeObjectives[];
  objectiveSpace?: {
    accuracy: { min: number; max: number };
    latency: { min: number; max: number };
    modelSize: { min: number; max: number };
  };
  
  // NEW: Performance stats
  totalTrainingMs?: number;
  totalInferenceMs?: number;
  avgSamplesPerSec?: number;
}
```

### Шаг 4: Расширить EvolutionSettings store

В `src/features/evolution-manager/model/store.ts`, расширить `EvolutionSettings`:

```typescript
export interface EvolutionSettings {
  // ... existing fields ...
  
  // Performance & Profiling
  profilingEnabled: boolean;
  memorySafetyMarginMb: number;
  estimatorSafetyFactor: number;
  
  // Multi-Objective
  mobjEnabled: boolean;
  primaryObjective: "accuracy";
  secondaryObjectives: Array<"latency" | "model_size" | "training_time" | "energy">;
  
  // Device Targeting
  deviceProfileId: string;
  isCustomDevice: boolean;
  customDeviceParams?: {
    ram_mb: number;
    vram_mb?: number;
    latency_budget_ms: number;
    max_model_size_mb?: number;
  };
  
  // Stopping Criteria
  stoppingPolicy: StoppingPolicy;
  
  // Genealogy
  genealogyTrackingEnabled: boolean;
  
  // Hidden Library
  autoSaveToHiddenLibrary: boolean;
}
```

### Шаг 5: Расширить EvolutionState store

В том же файле, расширить `EvolutionState`:

```typescript
export interface EvolutionState {
  // ... existing fields ...
  
  // Pareto & Multi-Objective
  paretoHistory: Map<number, GenerationParetoFront>;
  currentParetoFront: GenomeObjectives[];
  
  // Genealogy
  genealogyTree?: Map<string, GenomeGeneology>;
  
  // Performance tracking
  generationProfilingStats: Map<number, {
    generation: number;
    totalTrainingMs: number;
    totalInferenceMs: number;
    avgSamplesPerSec: number;
    peakConcurrentVramMb: number;
    totalJobsCompleted: number;
    totalJobsFailed: number;
  }>;
  
  // Stopping criteria progress
  currentStoppingProgress: {
    generationsSoFar: number;
    elapsedSeconds: number;
    plateauPatience: number;
    bestAccuracySoFar: number;
    triggeredCriteria?: string[];
  };
  
  // Hidden library auto-save count
  hiddenLibraryGenomeCount: number;
}
```

### Шаг 6: Создать helper hook для Pareto tracking

Создать `src/shared/hooks/useParetoTracking.ts`:

```typescript
import { useCallback } from "react";
import { useCanvasGenomeStore } from "@/entities/canvas-genome";
import { GenomeObjectives, GenerationParetoFront } from "@/shared/lib/dtos";

export function useParetoTracking() {
  const store = useCanvasGenomeStore();

  const updatePareto = useCallback(
    (generation: number, pareto: GenerationParetoFront) => {
      // Update store with Pareto front
      // Implementation depends on actual store structure
    },
    []
  );

  const isDominated = useCallback((a: GenomeObjectives, b: GenomeObjectives): boolean => {
    return (
      b.accuracy >= a.accuracy &&
      b.inference_latency_ms <= a.inference_latency_ms &&
      b.model_size_mb <= a.model_size_mb &&
      !(
        a.accuracy === b.accuracy &&
        a.inference_latency_ms === b.inference_latency_ms &&
        a.model_size_mb === b.model_size_mb
      )
    );
  }, []);

  const computeParetoFront = useCallback(
    (genomes: GenomeObjectives[]): GenomeObjectives[] => {
      const front: GenomeObjectives[] = [];
      for (const candidate of genomes) {
        if (!front.some((member) => isDominated(candidate, member))) {
          // Remove dominated members
          front.splice(
            0,
            front.length,
            ...front.filter((m) => !isDominated(m, candidate))
          );
          front.push(candidate);
        }
      }
      return front;
    },
    [isDominated]
  );

  return { updatePareto, isDominated, computeParetoFront };
}
```

### Шаг 7: Создать helper hook для Genealogy

Создать `src/shared/hooks/useGenealogy.ts`:

```typescript
import { useCallback } from "react";
import { GenomeGeneology, MutationType } from "@/shared/lib/dtos";

export function useGenealogy() {
  const buildAncestralChain = useCallback(
    (genomeId: string, genealogyMap: Map<string, GenomeGeneology>): string[] => {
      const chain = [genomeId];
      let current = genealogyMap.get(genomeId);

      while (current && current.parent_ids.length > 0) {
        chain.push(current.parent_ids[0]);
        current = genealogyMap.get(current.parent_ids[0]);
      }

      return chain;
    },
    []
  );

  const hasCycles = useCallback(
    (genealogyMap: Map<string, GenomeGeneology>): boolean => {
      for (const [genomeId, _] of genealogyMap) {
        const visited = new Set<string>();
        let current = genomeId;

        while (current && !visited.has(current)) {
          visited.add(current);
          const geom = genealogyMap.get(current);
          if (!geom || geom.parent_ids.length === 0) break;
          current = geom.parent_ids[0];
        }

        if (current && visited.has(current)) {
          return true; // Cycle detected
        }
      }

      return false;
    },
    []
  );

  return { buildAncestralChain, hasCycles };
}
```

### Шаг 8: Создать helper hook для Stopping Criteria

Создать `src/shared/hooks/useStoppingCriteria.ts`:

```typescript
import { useCallback } from "react";
import { StoppingCriterionType } from "@/shared/lib/dtos";
import { GenerationSnapshot } from "@/features/evolution-studio/model/types";

export function useStoppingCriteria() {
  const checkGenerationLimit = useCallback(
    (criterion: StoppingCriterionType, generation: number): boolean => {
      if (criterion.type !== "GenerationLimit") return false;
      return generation >= criterion.max_generations;
    },
    []
  );

  const checkFitnessPlateau = useCallback(
    (
      criterion: StoppingCriterionType,
      history: GenerationSnapshot[],
      patience: number
    ): boolean => {
      if (criterion.type !== "FitnessPlateau") return false;

      if (history.length < criterion.patience_generations) return false;

      const recent = history.slice(-criterion.patience_generations);
      const bestInRecent = Math.max(...recent.map((g) => g.bestFitness));
      const prevBest = history[history.length - criterion.patience_generations - 1]?.bestFitness || 0;

      const improvement = (bestInRecent - prevBest) / (Math.abs(prevBest) + 1e-6);
      return improvement < criterion.improvement_threshold;
    },
    []
  );

  const checkTimeLimit = useCallback(
    (criterion: StoppingCriterionType, elapsedSeconds: number): boolean => {
      if (criterion.type !== "TimeLimit") return false;
      return elapsedSeconds >= criterion.max_seconds;
    },
    []
  );

  const checkTargetAccuracy = useCallback(
    (criterion: StoppingCriterionType, bestAccuracy: number): boolean => {
      if (criterion.type !== "TargetAccuracy") return false;
      return bestAccuracy >= criterion.threshold;
    },
    []
  );

  return {
    checkGenerationLimit,
    checkFitnessPlateau,
    checkTimeLimit,
    checkTargetAccuracy,
  };
}
```

### Шаг 9: Обновить TypeScript конфиг

Если нужно, убедиться что tsconfig.json включает `skipLibCheck: false` и `strict: true` для type safety.

---

## Критерии готовности

- ✅ Файл `src/shared/lib/dtos.ts` создан со всеми DTO типами
- ✅ `PopulatedGenome` расширен новыми полями
- ✅ `GenerationSnapshot` расширен Pareto + genealogy коды
- ✅ `EvolutionSettings` расширен device + stopping + profiling
- ✅ `EvolutionState` расширен tracking полями
- ✅ Хук `useParetoTracking` создан и реализован
- ✅ Хук `useGenealogy` создан и реализован
- ✅ Хук `useStoppingCriteria` создан с helper функциями
- ✅ Все файлы компилируются без ошибок
- ✅ Типы экспортированы из `src/shared/index.ts`

---

## Тесты

Создать `src/shared/lib/dtos.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { GenomeObjectives } from "./dtos";

describe("DTOs", () => {
  it("should create GenomeObjectives", () => {
    const obj: GenomeObjectives = {
      genome_id: "g1",
      accuracy: 0.95,
      inference_latency_ms: 50,
      model_size_mb: 10,
      training_time_ms: 5000,
      is_dominated: false,
      domination_count: 0,
    };
    expect(obj.accuracy).toBe(0.95);
  });
});
```

Создать `src/shared/hooks/useGenealogy.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useGenealogy } from "./useGenealogy";
import { GenomeGeneology } from "../lib/dtos";

describe("useGenealogy", () => {
  it("should build ancestral chain", () => {
    const { result } = renderHook(() => useGenealogy());
    
    const genealogyMap = new Map<string, GenomeGeneology>([
      ["g3", { genome_id: "g3", parent_ids: ["g2"], generation: 2, /* ... */ } as any],
      ["g2", { genome_id: "g2", parent_ids: ["g1"], generation: 1, /* ... */ } as any],
      ["g1", { genome_id: "g1", parent_ids: [], generation: 0, /* ... */ } as any],
    ]);

    const chain = result.current.buildAncestralChain("g3", genealogyMap);
    expect(chain).toEqual(["g3", "g2", "g1"]);
  });

  it("should detect cycles in genealogy", () => {
    const { result } = renderHook(() => useGenealogy());
    
    const genealogyMap = new Map<string, GenomeGeneology>([
      ["g1", { genome_id: "g1", parent_ids: ["g2"], /* ... */ } as any],
      ["g2", { genome_id: "g2", parent_ids: ["g1"], /* ... */ } as any],
    ]);

    const hasCycle = result.current.hasCycles(genealogyMap);
    expect(hasCycle).toBe(true);
  });
});
```

Запустить:
```bash
npm run test -- src/shared/
```

---

## Вывод

- **Файлы**:
  - `src/shared/lib/dtos.ts` (новый)
  - `src/shared/hooks/useParetoTracking.ts` (новый)
  - `src/shared/hooks/useGenealogy.ts` (новый)
  - `src/shared/hooks/useStoppingCriteria.ts` (новый)
  - Расширения в store files

- **Строк кода**: ~300 новых LOC (TypeScript)
- **Зависимость**: Все Phase 2 frontend tasks используют эти типы и hooks

---

## Примечания

- Все типы должны быть exported из `src/shared/index.ts`
- Hooks используют React best practices (useCallback для memoization)
- Типы соответствуют Rust DTOs для лучшей интеграции
- Genealogy хук может быть расширен для визуализации после создания компонентов
