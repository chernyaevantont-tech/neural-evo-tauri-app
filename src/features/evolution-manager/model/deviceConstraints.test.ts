import { describe, expect, it } from 'vitest';
import {
    evaluateGenomeFeasibility,
    validateDeviceConstraintParams,
    type DeviceConstraintParams,
} from './deviceConstraints';
import type { GenomeObjectives } from '../../../shared/lib';

const constraints: DeviceConstraintParams = {
    mops_budget: 800,
    ram_mb: 8,
    flash_mb: 32,
    latency_budget_ms: 50,
};

const objective: GenomeObjectives = {
    genome_id: 'g-1',
    accuracy: 0.91,
    inference_latency_ms: 40,
    model_size_mb: 2,
    training_time_ms: 1500,
    is_dominated: false,
    domination_count: 0,
};

describe('deviceConstraints helpers', () => {
    it('validates positive numeric constraints', () => {
        const result = validateDeviceConstraintParams(constraints);
        expect(result.isValid).toBe(true);
        expect(result.fieldErrors).toEqual({});
    });

    it('returns errors for invalid values and warning for too low limits', () => {
        const result = validateDeviceConstraintParams({
            mops_budget: -1,
            ram_mb: 1,
            flash_mb: 0,
            latency_budget_ms: 2,
        });

        expect(result.isValid).toBe(false);
        expect(result.fieldErrors.mops_budget).toBeTruthy();
        expect(result.fieldErrors.flash_mb).toBeTruthy();
        expect(result.warnings.length).toBeGreaterThan(0);
    });

    it('computes feasibility and violation score', () => {
        const feasible = evaluateGenomeFeasibility(objective, constraints);
        expect(feasible.isFeasible).toBe(true);
        expect(feasible.violationScore).toBe(0);

        const infeasible = evaluateGenomeFeasibility(
            {
                ...objective,
                inference_latency_ms: 200,
                model_size_mb: 64,
            },
            constraints,
        );

        expect(infeasible.isFeasible).toBe(false);
        expect(infeasible.violationScore).toBeGreaterThan(0);
    });
});
