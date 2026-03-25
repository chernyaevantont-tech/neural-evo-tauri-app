import { describe, expect, it } from 'vitest';
import {
    getCriterionDescription,
    isStoppingPolicyValid,
    validateSingleCriterion,
    validateStoppingCriteria,
} from './stoppingCriteriaValidator';

describe('stoppingCriteriaValidator', () => {
    it('validates required positive limits', () => {
        expect(validateSingleCriterion({ type: 'GenerationLimit', max_generations: 0 })).toBe(
            'Max generations must be greater than 0',
        );
        expect(validateSingleCriterion({ type: 'TimeLimit', max_seconds: -1 })).toBe(
            'Time limit must be greater than 0 seconds',
        );
        expect(
            validateSingleCriterion({
                type: 'FitnessPlateau',
                patience_generations: 0,
                improvement_threshold: 0.001,
                monitor: 'best_fitness',
            }),
        ).toBe('Patience must be greater than 0');
    });

    it('validates target accuracy bounds [0, 1]', () => {
        expect(validateSingleCriterion({ type: 'TargetAccuracy', threshold: -0.1 })).toBe(
            'Target accuracy must be between 0 and 1',
        );
        expect(validateSingleCriterion({ type: 'TargetAccuracy', threshold: 1.1 })).toBe(
            'Target accuracy must be between 0 and 1',
        );
        expect(validateSingleCriterion({ type: 'TargetAccuracy', threshold: 0.9 })).toBeNull();
    });

    it('rejects duplicate ManualStop in full policy', () => {
        const error = validateStoppingCriteria(
            [{ type: 'ManualStop' }, { type: 'ManualStop' }],
            'any',
        );
        expect(error).toBe('Only one ManualStop criterion is allowed');
    });

    it('returns boolean helper state', () => {
        expect(
            isStoppingPolicyValid([{ type: 'GenerationLimit', max_generations: 100 }], 'any'),
        ).toBe(true);
        expect(
            isStoppingPolicyValid([{ type: 'GenerationLimit', max_generations: 0 }], 'any'),
        ).toBe(false);
    });

    it('builds readable criterion descriptions', () => {
        expect(
            getCriterionDescription({ type: 'GenerationLimit', max_generations: 150 }),
        ).toContain('150 generations');
        expect(getCriterionDescription({ type: 'TargetAccuracy', threshold: 0.88 })).toContain(
            '88.0%',
        );
    });
});
