import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useStoppingCriteria } from './useStoppingCriteria';

describe('useStoppingCriteria', () => {
    it('checks generation limit and time limit criteria', () => {
        const { result } = renderHook(() => useStoppingCriteria());

        expect(
            result.current.checkGenerationLimit(
                { type: 'GenerationLimit', max_generations: 10 },
                10,
            ),
        ).toBe(true);
        expect(
            result.current.checkGenerationLimit(
                { type: 'GenerationLimit', max_generations: 10 },
                9,
            ),
        ).toBe(false);

        expect(result.current.checkTimeLimit({ type: 'TimeLimit', max_seconds: 120 }, 121)).toBe(true);
        expect(result.current.checkTimeLimit({ type: 'TimeLimit', max_seconds: 120 }, 90)).toBe(false);
    });

    it('checks target accuracy with normalized score threshold', () => {
        const { result } = renderHook(() => useStoppingCriteria());

        expect(
            result.current.checkTargetAccuracy(
                { type: 'TargetAccuracy', threshold: 0.9 },
                0.91,
            ),
        ).toBe(true);
        expect(
            result.current.checkTargetAccuracy(
                { type: 'TargetAccuracy', threshold: 0.9 },
                0.89,
            ),
        ).toBe(false);
    });

    it('detects fitness plateau only when relative improvement is below threshold', () => {
        const { result } = renderHook(() => useStoppingCriteria());

        const criterion = {
            type: 'FitnessPlateau' as const,
            patience_generations: 3,
            improvement_threshold: 0.03,
            monitor: 'best_fitness' as const,
        };

        const plateauHistory = [
            { bestFitness: 1.0 },
            { bestFitness: 1.005 },
            { bestFitness: 1.006 },
            { bestFitness: 1.007 },
        ];

        const improvingHistory = [
            { bestFitness: 1.0 },
            { bestFitness: 1.02 },
            { bestFitness: 1.04 },
            { bestFitness: 1.06 },
        ];

        expect(result.current.checkFitnessPlateau(criterion, plateauHistory)).toBe(true);
        expect(result.current.checkFitnessPlateau(criterion, improvingHistory)).toBe(false);
    });
});