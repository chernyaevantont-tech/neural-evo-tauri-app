# Zero-Cost Proxies Integration Guide

## Реализован первый вариант (Two-Stage)

### ✅ Завершено

1. **Rust модуль** (`src-tauri/src/zero_cost_proxies.rs`)
   - `ZeroCostConfig` — конфигурация
   - `ZeroCostMetrics` — результаты метрик
   - `StrategyDecision` — enum для решений
   - Принятие решений на основе synflow score

2. **TypeScript хук** (`src/features/evolution-studio/model/useZeroCostEvaluation.ts`)
   - `computeZeroCostScore()` — вычисление через Tauri
   - `getRecommendedEpochs()` — получение рекомендаций по эпохам
   - `calculateHybridFitness()` — комбинация proxy + accuracy
   - `estimateTimeSavings()` — оценка экономии времени

3. **Store обновлен** (`src/features/evolution-manager/model/store.ts`)
   - `useZeroCostProxies: boolean`
   - `zeroCostStrategy: 'two-stage' | 'early-stopping'`
   - `fastPassThreshold: number` (0-1)
   - `partialTrainingEpochs: number`

4. **UI контролы** (`src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx`)
   - Чекбокс для включения
   - Выбор стратегии (Two-Stage / Early Stopping)
   - Слайдер для threshold
   - Слайдер для partial epochs

5. **CSS стили** (`EvolutionSettingsPanel.module.css`)
   - `.helperText` — вспомогательный текст
   - `.radioGroup` — группа radio buttons
   - `.radioLabel` — стиль label для radio buttons

---

## 🔧 Интеграция в useEvolutionLoop (Следующий шаг)

### Структура Two-Stage Evaluation

```
Для каждой архитектуры в популяции:

├─ Шаг 1: Zero-Cost Score
│  ├─ Forward pass + Backward pass (1 батч)
│  ├─ Вычислить SynFlow метрику
│  └─ Нормализовать в 0-1 диапазон
│
└─ Шаг 2: Решение по обучению
   ├─ if (score > fastPassThreshold) → Full Training (50 эпох)
   ├─ else if (score > 0.3) → Partial Training (20 эпох)
   └─ else → Skip Training (нет обучения)
```

### Код Интеграции

В `useEvolutionLoop.ts` в месте обучения генома:

```typescript
import { computeZeroCostScore, getRecommendedEpochs, calculateHybridFitness, ZeroCostMetrics } from './useZeroCostEvaluation';

// Получить настройки
const settings = useEvolutionSettingsStore.getState();

// Для каждого генома
for (const genomePopulated of generation) {
    let zeroCostMetrics: ZeroCostMetrics | null = null;
    let trainEpochs = settings.evalEpochs;
    
    // Шаг 1: Если включены zero-cost proxies, вычислить score
    if (settings.useZeroCostProxies) {
        try {
            zeroCostMetrics = await computeZeroCostScore(
                JSON.stringify(genomeJSON),
                JSON.stringify(trainBatchSample),
                {
                    enabled: true,
                    strategy: settings.zeroCostStrategy,
                    fastPassThreshold: settings.fastPassThreshold,
                    partialTrainingEpochs: settings.partialTrainingEpochs,
                    useVoting: false, // Пока только SynFlow
                }
            );
            
            // Получить рекомендованное количество эпох
            const recommended = getRecommendedEpochs(
                zeroCostMetrics,
                settings.evalEpochs,
                settings.partialTrainingEpochs
            );
            
            if (recommended === null) {
                // Пропустить обучение
                genomePopulated.adjustedFitness = zeroCostMetrics.normalized_score;
                continue;
            }
            
            trainEpochs = recommended;
        } catch (error) {
            console.warn('Zero-cost scoring failed, falling back to full training', error);
            // Продолжить с полным обучением
        }
    }
    
    // Шаг 2: Обучить на рекомендованное количество эпох
    const result = await invoke<EvaluationResult>('train_and_evaluate_genome', {
        genomeJSON: JSON.stringify(genomeJSON),
        epochs: trainEpochs,
        batchSize: settings.batchSize,
        // ... остальные параметры
    });
    
    // Шаг 3: Вычислить итоговую фитнес-функцию
    if (zeroCostMetrics) {
        genomePopulated.adjustedFitness = calculateHybridFitness(
            zeroCostMetrics.normalized_score,
            result.accuracy,
            {
                enabled: true,
                strategy: settings.zeroCostStrategy,
                fastPassThreshold: settings.fastPassThreshold,
                partialTrainingEpochs: settings.partialTrainingEpochs,
                useVoting: false,
            }
        );
    } else {
        genomePopulated.adjustedFitness = result.accuracy;
    }
}
```

---

## 📊 Два-Этапная Логика

### Сценарий 1: Архитектура с ВЫСОКИм SynFlow (0.8)

```
Zero-Cost Score: 0.8
FastPassThreshold: 0.6

Решение: FULL TRAINING ✓

Процесс:
├─ Zero-cost: 1 мс (скан через сеть)
├─ Full training: 50 эпох × 10 мин = 500 мин
└─ Финальная fitness = 0.7 × accuracy + 0.3 × 0.8

Результат: Многообещающая архитектура проходит полное обучение
Время потрачено: 500 + 1 = 501 мин
```

### Сценарий 2: Архитектура со СРЕДНИМ SynFlow (0.5)

```
Zero-Cost Score: 0.5
FastPassThreshold: 0.6

Решение: PARTIAL TRAINING ⚠️

Процесс:
├─ Zero-cost: 1 мс (скан через сеть)
├─ Partial training: 20 эпох × 10 мин = 200 мин
└─ Финальная fitness = 0.7 × accuracy_20 + 0.3 × 0.5

Результат: Неясная архитектура получает быстрое обучение
Время потрачено: 200 + 1 = 201 мин (60% экономия)
```

### Сценарий 3: Архитектура с НИЗКИм SynFlow (0.2)

```
Zero-Cost Score: 0.2
FastPassThreshold: 0.6

Решение: SKIP TRAINING ❌

Процесс:
├─ Zero-cost: 1 мс (скан через сеть)
├─ Training: NO
└─ Финальная fitness = 0.2 (используем proxy score)

Результат: Плохая архитектура отсеивается
Время потрачено: 1 мс (99.98% экономия)

Евоюция: Эта архитектура получит низкий ранг и редко будет выбрана
```

---

## 📈 Ожидаемое улучшение производительности

### До Zero-Cost Proxies

```
Поколение 50 генов × 50 эпох × 10 мин/эпоха
= 50 × 50 × 10 
= 25,000 минут (416 часов) на поколение
```

### После Zero-Cost Proxies (Two-Stage)

```
Вычисление эконом:
- 20% архитектур (10 штук): skip (1 мс каждая)
- 40% архитектур (20 штук): partial (20 эпох × 10 мин)
- 40% архитектур (20 штук): full (50 эпох × 10 мин)

Расчет:
├─ Zero-cost для всех 50: 50 × 0.001 мс ≈ 0 мин
├─ Skip: 10 архитектур × 0 мин = 0 мин
├─ Partial: 20 архитектур × 200 мин = 4,000 мин
└─ Full: 20 архитектур × 500 мин = 10,000 мин

Итого: 14,000 минут (233 часов) на поколение
Ускорение: 25,000 / 14,000 = 1.78x (или 44% сокращение времени)
```

⚠️ **Примечание:** В предыдущем примере моя оценка была оптимистична.
Реальная оценка для Two-Stage — **1.5-2x ускорение** (масло-масло, не забываем zero-cost overhead).

Для **Early-Stopping стратегии** будет **2-3x ускорение**.

---

## 🚀 Дополнительные Улучшения

### 1. Добавить Логирование Zero-Cost Scores

```typescript
const log = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    addLogEntry({
        time: timestamp,
        message: msg,
        type: 'info',
    });
};

// Когда вычисляется score:
log(`Genome ${id}: SynFlow=${zeroCostMetrics?.synflow.toFixed(2)} → Decision: ${zeroCostMetrics?.strategy_decision}`);
```

### 2. Отслеживать Экономию Времени

```typescript
let totalTimeSkipped = 0;
let skippedCount = 0;

if (zeroCostMetrics?.strategy_decision === 'skip') {
    skippedCount++;
    totalTimeSkipped += settings.evalEpochs * 10; // примерное время
}

// В конце поколения:
const speedup = estimateTimeSavings(populationSize, 500, avgScore, settings);
log(`Generation ${gen}: Skipped ${skippedCount} genomes, saved ~${speedup.savedTime} min (${speedup.speedup.toFixed(1)}x speedup)`);
```

### 3. Адаптивный Threshold

```typescript
// Если первые 10 поколений показывают плохие результаты со skipped генами,
// повышаем threshold
if (skippedCount > populationSize * 0.6 && avgAccuracy < 0.7) {
    settings.setFastPassThreshold(settings.fastPassThreshold + 0.1);
    log(`Adaptive: Increased fastPassThreshold to ${settings.fastPassThreshold}`);
}
```

---

## ⚠️ Потенциальные Проблемы и Решения

### Проблема 1: SynFlow Score Неправильно Коррелирует

**Признак:** Архитектуры с высоким score показывают низкую accuracy

**Решение:**
- Убедитесь, что нормализация правильная (делим на 10)
- Попробуйте Early-Stopping стратегию вместо Two-Stage
- Увеличьте `fastPassThreshold` (более консервативно)

### Проблема 2: Computation Overhead

**Признак:** Даже с пропуском обучения эволюция медленнее

**Решение:**
- Zero-cost вычисления могут быть дорогими на CPU
- Убедитесь, что используется GPU для forward/backward passes
- Кэшируйте градиенты если возможно

### Проблема 3: Недостаток Разнообразия

**Признак:** Популяция сходится слишком быстро

**Решение:**
- Используйте больший `randomInitRatio` для большего разнообразия
- Добавьте "rescue mutation" для低-scoring генов
- Явно избегайте обучения только лучших (培养проигравших)

---

## 📝 Чеклист для Полной Реализации

- [x] Создать `zero_cost_proxies.rs` модуль
- [x] Добавить DTOs для конфигурации
- [x] Создать TypeScript хук `useZeroCostEvaluation`
- [x] Обновить Zustand store
- [x] Добавить UI контролы
- [ ] Интегрировать в `useEvolutionLoop` (требует changes в evolution loop structure)
- [ ] Добавить логирование zero-cost scores
- [ ] Тестировать на реальных датасетах
- [ ] Оптимизировать computation overhead на Rust стороне
- [ ] Добавить adaptive threshold механизм

---

## 🔗 Связанные Файлы

- **Rust модуль:** `src-tauri/src/zero_cost_proxies.rs`
- **DTO:** `src-tauri/src/dtos.rs` (добавлен `ZeroCostConfigDto`)
- **TypeScript хук:** `src/features/evolution-studio/model/useZeroCostEvaluation.ts` (новый)
- **Store:** `src/features/evolution-manager/model/store.ts` (обновлен)
- **UI Panel:** `src/pages/evolution-studio-page/EvolutionSettingsPanel.tsx` (обновлен)
- **Styles:** `EvolutionSettingsPanel.module.css` (добавлены стили)

---

## 📚 Документация

- **Zero-Cost Proxies Guide:** [ZERO_COST_PROXIES.md](../ZERO_COST_PROXIES.md)
- **Оригинальная статья:** https://arxiv.org/abs/2101.08134
- **GitHub реализация:** https://github.com/mohsaied/zero-cost-nas
