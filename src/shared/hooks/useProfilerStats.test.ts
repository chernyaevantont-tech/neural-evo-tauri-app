import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { PopulatedGenome } from '../../entities/genome';
import { useProfilerStats } from './useProfilerStats';

function makeGenome(id: string, trainMs: number, inferMs: number, memoryMb: number, samplesPerSec: number): PopulatedGenome {
    return {
        id,
        genome: {} as any,
        nodes: [],
        profiler: {
            train_start_ms: 0,
            first_batch_ms: 10,
            train_end_ms: trainMs,
            total_train_duration_ms: trainMs,
            val_start_ms: 0,
            val_end_ms: 0,
            val_duration_ms: 0,
            test_start_ms: 0,
            test_end_ms: 0,
            test_duration_ms: 0,
            peak_active_memory_mb: memoryMb,
            peak_model_params_mb: 0,
            peak_gradient_mb: 0,
            peak_optim_state_mb: 0,
            peak_activation_mb: 0,
            samples_per_sec: samplesPerSec,
            inference_msec_per_sample: inferMs,
            batch_count: 1,
        },
    };
}

describe('useProfilerStats', () => {
    it('computes averages and extremes from profiler data', () => {
        const genomes = [
            makeGenome('g1', 1000, 0.5, 100, 40),
            makeGenome('g2', 2000, 1.0, 200, 20),
            makeGenome('g3', 500, 0.25, 150, 60),
        ];

        const { result } = renderHook(() => useProfilerStats(genomes));

        expect(result.current.avgTrainingTime).toBeCloseTo(1166.67, 1);
        expect(result.current.avgInferenceLatency).toBeCloseTo(0.583, 2);
        expect(result.current.totalMemory).toBe(450);
        expect(result.current.avgThroughput).toBeCloseTo(40, 4);
        expect(result.current.fastestGenome?.id).toBe('g3');
        expect(result.current.slowestGenome?.id).toBe('g2');
    });

    it('returns zeros for empty arrays', () => {
        const { result } = renderHook(() => useProfilerStats([]));

        expect(result.current.avgTrainingTime).toBe(0);
        expect(result.current.avgInferenceLatency).toBe(0);
        expect(result.current.totalMemory).toBe(0);
        expect(result.current.avgThroughput).toBe(0);
        expect(result.current.fastestGenome).toBeUndefined();
        expect(result.current.slowestGenome).toBeUndefined();
    });
});
