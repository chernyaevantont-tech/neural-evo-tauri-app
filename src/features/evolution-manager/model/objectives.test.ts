import { describe, expect, it } from 'vitest';
import {
    normalizeObjectiveWeights,
    normalizeSecondaryObjectives,
    validateObjectives,
} from './objectives';

describe('objectives helpers', () => {
    it('validates multi-objective requires at least one secondary objective', () => {
        expect(validateObjectives('multi', [])).toContain('requires at least one secondary objective');
        expect(validateObjectives('multi', ['latency'])).toBeNull();
        expect(validateObjectives('single', [])).toBeNull();
    });

    it('normalizes and orders secondary objectives deterministically', () => {
        expect(normalizeSecondaryObjectives(['model_size', 'latency', 'model_size'])).toEqual([
            'latency',
            'model_size',
        ]);
    });

    it('normalizes active objective weights to sum 1', () => {
        const normalized = normalizeObjectiveWeights(
            {
                accuracy: 2,
                latency: 1,
                model_size: 1,
                train_time: 0,
            },
            ['latency', 'model_size'],
        );

        const sum = normalized.accuracy + normalized.latency + normalized.model_size + normalized.train_time;
        expect(sum).toBeCloseTo(1, 6);
        expect(normalized.train_time).toBe(0);
    });
});
