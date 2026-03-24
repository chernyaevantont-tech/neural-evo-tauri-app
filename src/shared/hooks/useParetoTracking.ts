import { useCallback } from 'react';
import type { GenerationParetoFront, GenomeObjectives } from '../lib/dtos';

export function useParetoTracking() {
    const updatePareto = useCallback(
        (
            generation: number,
            pareto: GenerationParetoFront,
            onUpdate?: (generation: number, pareto: GenerationParetoFront) => void,
        ) => {
            onUpdate?.(generation, pareto);
        },
        [],
    );

    const isDominated = useCallback((a: GenomeObjectives, b: GenomeObjectives): boolean => {
        const hasStrictBetter =
            b.accuracy > a.accuracy ||
            b.inference_latency_ms < a.inference_latency_ms ||
            b.model_size_mb < a.model_size_mb;

        return (
            b.accuracy >= a.accuracy &&
            b.inference_latency_ms <= a.inference_latency_ms &&
            b.model_size_mb <= a.model_size_mb &&
            hasStrictBetter
        );
    }, []);

    const computeParetoFront = useCallback(
        (genomes: GenomeObjectives[]): GenomeObjectives[] => {
            const front: GenomeObjectives[] = [];
            for (const candidate of genomes) {
                if (!front.some((member) => isDominated(candidate, member))) {
                    front.splice(
                        0,
                        front.length,
                        ...front.filter((member) => !isDominated(member, candidate)),
                    );
                    front.push(candidate);
                }
            }
            return front;
        },
        [isDominated],
    );

    return { updatePareto, isDominated, computeParetoFront };
}
