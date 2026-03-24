import { useCallback } from 'react';
import type { StoppingCriterionType } from '../lib/dtos';

interface GenerationLike {
    bestFitness: number;
}

export function useStoppingCriteria() {
    const checkGenerationLimit = useCallback(
        (criterion: StoppingCriterionType, generation: number): boolean => {
            if (criterion.type !== 'GenerationLimit') return false;
            return generation >= criterion.max_generations;
        },
        [],
    );

    const checkFitnessPlateau = useCallback(
        (criterion: StoppingCriterionType, history: GenerationLike[]): boolean => {
            if (criterion.type !== 'FitnessPlateau') return false;
            if (history.length <= criterion.patience_generations) return false;

            const recent = history.slice(-criterion.patience_generations);
            const bestInRecent = Math.max(...recent.map((g) => g.bestFitness));
            const prevBest = history[history.length - criterion.patience_generations - 1]?.bestFitness ?? 0;
            const improvement = (bestInRecent - prevBest) / (Math.abs(prevBest) + 1e-6);

            return improvement < criterion.improvement_threshold;
        },
        [],
    );

    const checkTimeLimit = useCallback(
        (criterion: StoppingCriterionType, elapsedSeconds: number): boolean => {
            if (criterion.type !== 'TimeLimit') return false;
            return elapsedSeconds >= criterion.max_seconds;
        },
        [],
    );

    const checkTargetAccuracy = useCallback(
        (criterion: StoppingCriterionType, bestAccuracy: number): boolean => {
            if (criterion.type !== 'TargetAccuracy') return false;
            return bestAccuracy >= criterion.threshold;
        },
        [],
    );

    return {
        checkGenerationLimit,
        checkFitnessPlateau,
        checkTimeLimit,
        checkTargetAccuracy,
    };
}
