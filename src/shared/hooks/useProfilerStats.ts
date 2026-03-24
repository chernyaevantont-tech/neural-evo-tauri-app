import { useMemo } from 'react';
import type { PopulatedGenome } from '../../entities/genome';

interface ProfilerStatsResult {
    avgTrainingTime: number;
    avgInferenceLatency: number;
    totalMemory: number;
    avgThroughput: number;
    slowestGenome?: PopulatedGenome;
    fastestGenome?: PopulatedGenome;
}

export function useProfilerStats(genomes: PopulatedGenome[]): ProfilerStatsResult {
    return useMemo(() => {
        if (genomes.length === 0) {
            return {
                avgTrainingTime: 0,
                avgInferenceLatency: 0,
                totalMemory: 0,
                avgThroughput: 0,
                slowestGenome: undefined,
                fastestGenome: undefined,
            };
        }

        const avgTrainingTime =
            genomes.reduce((sum, g) => sum + (g.profiler?.total_train_duration_ms ?? 0), 0) /
            genomes.length;

        const avgInferenceLatency =
            genomes.reduce((sum, g) => sum + (g.profiler?.inference_msec_per_sample ?? 0), 0) /
            genomes.length;

        const totalMemory = genomes.reduce(
            (sum, g) => sum + (g.profiler?.peak_active_memory_mb ?? 0),
            0,
        );

        const avgThroughput =
            genomes.reduce((sum, g) => sum + (g.profiler?.samples_per_sec ?? 0), 0) /
            genomes.length;

        const byTrainingTime = [...genomes].sort(
            (a, b) =>
                (a.profiler?.total_train_duration_ms ?? Number.POSITIVE_INFINITY) -
                (b.profiler?.total_train_duration_ms ?? Number.POSITIVE_INFINITY),
        );

        const fastestGenome = byTrainingTime.find((g) => g.profiler?.total_train_duration_ms !== undefined);
        const slowestGenome =
            [...byTrainingTime]
                .reverse()
                .find((g) => g.profiler?.total_train_duration_ms !== undefined) ?? fastestGenome;

        return {
            avgTrainingTime,
            avgInferenceLatency,
            totalMemory,
            avgThroughput,
            slowestGenome,
            fastestGenome,
        };
    }, [genomes]);
}
