# Задача 102: Performance Profiler Frontend

**Фаза**: 2 (Core Features - Metrics)  
**Сложность**: Low  
**Время**: 4 часа  
**Зависимости**: Task 101, Task 003  
**Выполнит**: Frontend разработчик

---

## Описание

Обновить frontend для отображения profiler data: новые столбцы в таблице поколений (training time, inference latency, peak memory), детальный модаль с breakdown по train/val/test.

---

## Входные данные

- `src/shared/lib/dtos.ts` (TrainingProfiler interface из Task 003)
- Существующие компоненты эволюции (таблица поколений, детали генома)

---

## Пошаговое выполнение

### Шаг 1: Обновить GenerationStatsTable компонент

Расширить таблицу поколений в `src/features/evolution-studio/ui/GenerationStatsTable.tsx`:

```typescript
export function GenerationStatsTable({ generations }: { generations: GenerationSnapshot[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Gen</th>
          <th>Genomes</th>
          <th>Best Fitness</th>
          <th>Avg Nodes</th>
          {/* NEW: */}
          <th>Training Time (ms)</th>
          <th>Avg Inference (ms)</th>
          <th>Peak VRAM (MB)</th>
          <th>Throughput (samples/s)</th>
        </tr>
      </thead>
      <tbody>
        {generations.map((gen) => (
          <tr key={gen.generation}>
            <td>{gen.generation}</td>
            <td>{gen.genomes.length}</td>
            <td>{gen.bestFitness.toFixed(3)}</td>
            <td>{gen.avgNodes.toFixed(1)}</td>
            {/* NEW: */}
            <td>{gen.totalTrainingMs?.toLocaleString()}</td>
            <td>{(gen.totalInferenceMs ?? 0 / (gen.genomes.length || 1)).toFixed(2)}</td>
            <td>{(gen.genomes[0]?.profiler?.peak_active_memory_mb ?? 0).toFixed(0)}</td>
            <td>{gen.avgSamplesPerSec?.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

### Шаг 2: Создать GenomeProfilerModal компонент

Создать `src/features/evolution-studio/ui/GenomeProfilerModal.tsx`:

```typescript
import { TrainingProfiler } from "@/shared/lib/dtos";
import React from "react";

interface GenomeProfilerModalProps {
  genome_id: string;
  profiler: TrainingProfiler;
  onClose: () => void;
}

export function GenomeProfilerModal({ genome_id, profiler, onClose }: GenomeProfilerModalProps) {
  const formatMs = (ms: number) => `${(ms / 1000).toFixed(2)}s`;
  const formatMb = (mb: number) => `${mb.toFixed(1)}MB`;

  return (
    <div className="modal">
      <div className="modal-content">
        <h2>Profiler: {genome_id}</h2>
        
        {/* Training Section */}
        <section>
          <h3>Training Phase</h3>
          <dl>
            <dt>Duration:</dt>
            <dd>{formatMs(profiler.total_train_duration_ms)}</dd>
            
            <dt>First Batch Latency:</dt>
            <dd>{formatMs(profiler.first_batch_ms)}</dd>
            
            <dt>Batch Count:</dt>
            <dd>{profiler.batch_count}</dd>
            
            <dt>Throughput:</dt>
            <dd>{profiler.samples_per_sec.toFixed(1)} samples/sec</dd>
            
            <dt>Early Stop Epoch:</dt>
            <dd>{profiler.early_stop_epoch ?? "N/A"}</dd>
          </dl>
        </section>

        {/* Validation Section */}
        <section>
          <h3>Validation Phase</h3>
          <dl>
            <dt>Duration:</dt>
            <dd>{formatMs(profiler.val_duration_ms)}</dd>
            
            <dt>Inference Latency (per sample):</dt>
            <dd>{profiler.inference_msec_per_sample.toFixed(3)}ms</dd>
          </dl>
        </section>

        {/* Test Section */}
        <section>
          <h3>Test Phase</h3>
          <dl>
            <dt>Duration:</dt>
            <dd>{formatMs(profiler.test_duration_ms)}</dd>
          </dl>
        </section>

        {/* Memory Breakdown */}
        <section>
          <h3>Memory Peaks</h3>
          <dl>
            <dt>Total Active:</dt>
            <dd>{formatMb(profiler.peak_active_memory_mb)}</dd>
            
            <dt>Model Params:</dt>
            <dd>{formatMb(profiler.peak_model_params_mb)}</dd>
            
            <dt>Gradients:</dt>
            <dd>{formatMb(profiler.peak_gradient_mb)}</dd>
            
            <dt>Optimizer State:</dt>
            <dd>{formatMb(profiler.peak_optim_state_mb)}</dd>
            
            <dt>Activations:</dt>
            <dd>{formatMb(profiler.peak_activation_mb)}</dd>
          </dl>
        </section>

        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
```

### Шаг 3: Обновить GenomeDetailView

В компоненте `src/features/genome-library/ui/GenomeDetailPanel.tsx`, добавить кнопку для открытия профилера:

```typescript
export function GenomeDetailPanel({ genome }: { genome: PopulatedGenome }) {
  const [showProfiler, setShowProfiler] = React.useState(false);

  return (
    <div className="detail-panel">
      {/* ... existing fields ... */}
      
      {genome.profiler && (
        <>
          <button onClick={() => setShowProfiler(true)}>View Profiler Details</button>
          {showProfiler && (
            <GenomeProfilerModal
              genome_id={genome.id}
              profiler={genome.profiler}
              onClose={() => setShowProfiler(false)}
            />
          )}
        </>
      )}
    </div>
  );
}
```

### Шаг 4: Создать ComparisonChart компонент

Создать `src/widgets/genome-comparison/ComparisonCharts.tsx`:

```typescript
import React from "react";
import { Scatter } from "react-chartjs-2";
import { PopulatedGenome } from "@/entities/canvas-genome";

interface ComparisonChartsProps {
  genomes: PopulatedGenome[];
}

export function ComparisonCharts({ genomes }: ComparisonChartsProps) {
  const accuracyVsTime = genomes.map((g) => ({
    x: g.profiler?.total_train_duration_ms ?? 0,
    y: g.objectives?.accuracy ?? 0,
    label: g.id,
  }));

  const accuracyVsMemory = genomes.map((g) => ({
    x: g.profiler?.peak_active_memory_mb ?? 0,
    y: g.objectives?.accuracy ?? 0,
    label: g.id,
  }));

  return (
    <div className="comparison-charts">
      <div className="chart">
        <h4>Accuracy vs Training Time</h4>
        <Scatter
          data={{
            datasets: [
              {
                label: "Genomes",
                data: accuracyVsTime,
                backgroundColor: "rgba(75, 192, 192, 0.6)",
              },
            ],
          }}
          options={{
            scales: {
              x: { title: { display: true, text: "Training Time (ms)" } },
              y: { title: { display: true, text: "Accuracy" } },
            },
          }}
        />
      </div>

      <div className="chart">
        <h4>Accuracy vs Peak Memory</h4>
        <Scatter
          data={{
            datasets: [
              {
                label: "Genomes",
                data: accuracyVsMemory,
                backgroundColor: "rgba(192, 75, 192, 0.6)",
              },
            ],
          }}
          options={{
            scales: {
              x: { title: { display: true, text: "Peak Memory (MB)" } },
              y: { title: { display: true, text: "Accuracy" } },
            },
          }}
        />
      </div>
    </div>
  );
}
```

### Шаг 5: Добавить hook для профилер-related queries

Создать `src/shared/hooks/useProfilerStats.ts`:

```typescript
import { PopulatedGenome } from "@/entities/canvas-genome";

export function useProfilerStats(genomes: PopulatedGenome[]) {
  const avgTrainingTime = genomes.reduce((sum, g) => sum + (g.profiler?.total_train_duration_ms ?? 0), 0) / genomes.length;
  
  const avgInferenceLatency = genomes.reduce((sum, g) => sum + (g.profiler?.inference_msec_per_sample ?? 0), 0) / genomes.length;
  
  const totalMemory = genomes.reduce((sum, g) => sum + (g.profiler?.peak_active_memory_mb ?? 0), 0);
  
  const avgThroughput = genomes.reduce((sum, g) => sum + (g.profiler?.samples_per_sec ?? 0), 0) / genomes.length;

  return {
    avgTrainingTime,
    avgInferenceLatency,
    totalMemory,
    avgThroughput,
    slowestGenome: genomes.reduce((max, g) => (g.profiler?.total_train_duration_ms ?? 0) > (max.profiler?.total_train_duration_ms ?? 0) ? g : max),
    fastestGenome: genomes.reduce((min, g) => (g.profiler?.total_train_duration_ms ?? 0) < (min.profiler?.total_train_duration_ms ?? 0) ? g : min),
  };
}
```

### Шаг 6: Интегрировать в эволюционную dashboard

В `src/widgets/evolution-dashboard/EvolutionDashboard.tsx`, добавить профилер карточку:

```typescript
export function EvolutionDashboard() {
  const { profilerStats } = useEvolutionState();

  return (
    <div className="dashboard">
      {/* ... existing cards ... */}
      
      {/* NEW: Performance Metrics Card */}
      <card className="metric-card">
        <h3>Performance Metrics</h3>
        <dl>
          <dt>Avg Training Time:</dt>
          <dd>{(profilerStats.avgTrainingTime / 1000).toFixed(2)}s</dd>
          
          <dt>Avg Inference Latency:</dt>
          <dd>{profilerStats.avgInferenceLatency.toFixed(3)}ms/sample</dd>
          
          <dt>Avg Throughput:</dt>
          <dd>{profilerStats.avgThroughput.toFixed(1)} samples/sec</dd>
        </dl>
      </card>
    </div>
  );
}
```

---

## Критерии готовности

- ✅ GenerationStatsTable расширена новыми столбцами (4 новых)
- ✅ GenomeProfilerModal компонент создан и функционален
- ✅ GenomeDetailPanel обновлен с кнопкой открытия профилера
- ✅ ComparisonCharts компонент создан (2 scatter plot)
- ✅ useProfilerStats хук создан и экспортирован
- ✅ Интегрировано в EvolutionDashboard
- ✅ TypeScript компилируется без ошибок
- ✅ Нет runtime warnings

---

## Тесты

```typescript
import { render, screen } from "@testing-library/react";
import { GenomeProfilerModal } from "./GenomeProfilerModal";

describe("GenomeProfilerModal", () => {
  it("should display profiler information", () => {
    const profiler = {
      total_train_duration_ms: 5000,
      first_batch_ms: 100,
      batch_count: 100,
      // ... other fields
    };

    render(
      <GenomeProfilerModal
        genome_id="g1"
        profiler={profiler}
        onClose={() => {}}
      />
    );

    expect(screen.getByText(/Training Phase/)).toBeInTheDocument();
    expect(screen.getByText(/5.00s/)).toBeInTheDocument();
  });
});
```

Запустить: `npm run test -- src/features/evolution-studio/`

---

## Вывод

- **Компоненты**: GenomeProfilerModal, GenomeDetailPanel (update), ComparisonCharts
- **Hooks**: useProfilerStats
- **Строк кода**: ~250 новых LOC (TypeScript/React)
- **Зависимость**: Используется в Task 202 (evolution dashboard)

---

## Примечания

- Все timings в миллисекундах форматируются в секунды для читаемости
- Memory values in MB
- Профилер данные optional - UI gracefully handles missing data
- Chart.js required для scatter plots (должен быть уже установлен)
