import { renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { GenerationParetoFront, GenomeObjectives } from '../lib/dtos';
import { useParetoTracking } from './useParetoTracking';

function objective(
    genomeId: string,
    accuracy: number,
    latencyMs: number,
    modelSizeMb: number,
): GenomeObjectives {
    return {
        genome_id: genomeId,
        accuracy,
        inference_latency_ms: latencyMs,
        model_size_mb: modelSizeMb,
        training_time_ms: 100,
        is_dominated: false,
        domination_count: 0,
    };
}

describe('useParetoTracking', () => {
    it('calls update callback with generation and pareto payload', () => {
        const onUpdate = vi.fn();
        const pareto: GenerationParetoFront = {
            generation: 2,
            total_genomes: 1,
            pareto_members: [objective('g1', 0.9, 10, 1)],
            objectives_3d: [[0.9, 10, 1]],
        };

        const { result } = renderHook(() => useParetoTracking());
        result.current.updatePareto(2, pareto, onUpdate);

        expect(onUpdate).toHaveBeenCalledWith(2, pareto);
    });

    it('correctly detects strict domination and non-domination tradeoffs', () => {
        const { result } = renderHook(() => useParetoTracking());

        const weak = objective('weak', 0.8, 20, 2.2);
        const strong = objective('strong', 0.85, 18, 1.9);
        const tradeoff = objective('tradeoff', 0.92, 40, 1.2);

        expect(result.current.isDominated(weak, strong)).toBe(true);
        expect(result.current.isDominated(tradeoff, strong)).toBe(false);
    });

    it('returns only non-dominated members in computed pareto front', () => {
        const { result } = renderHook(() => useParetoTracking());

        const genomes = [
            objective('a', 0.8, 20, 2.0),
            objective('b', 0.82, 18, 1.8),
            objective('c', 0.9, 50, 1.0),
            objective('d', 0.81, 22, 2.2),
        ];

        const front = result.current.computeParetoFront(genomes);
        const ids = new Set(front.map((g) => g.genome_id));

        expect(ids.has('b')).toBe(true);
        expect(ids.has('c')).toBe(true);
        expect(ids.has('a')).toBe(false);
        expect(ids.has('d')).toBe(false);
    });
});