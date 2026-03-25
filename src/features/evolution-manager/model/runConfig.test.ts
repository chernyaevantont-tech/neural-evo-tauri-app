import { describe, expect, it } from 'vitest';
import { buildEvolutionRunConfig } from './runConfig';
import { useEvolutionSettingsStore } from './store';

function resetState() {
    useEvolutionSettingsStore.setState({
        mobjEnabled: true,
        secondaryObjectives: ['model_size', 'latency'],
        objectiveWeightsEnabled: true,
        objectiveWeights: {
            accuracy: 0.6,
            latency: 0.2,
            model_size: 0.2,
            train_time: 0,
        },
        customDeviceParams: {
            mops_budget: 1600,
            ram_mb: 64,
            flash_mb: 128,
            latency_budget_ms: 20,
            max_model_size_mb: 128,
        },
        stoppingPolicy: {
            policy_type: 'all',
            criteria: [
                { type: 'GenerationLimit', max_generations: 200 },
                { type: 'TargetAccuracy', threshold: 0.95 },
            ],
        },
        profilingEnabled: true,
        memorySafetyMarginMb: 256,
        estimatorSafetyFactor: 1.2,
        memoryMode: 'hybrid',
    });
}

describe('buildEvolutionRunConfig', () => {
    it('serializes config deterministically and keeps policy any/all', () => {
        resetState();
        const settings = useEvolutionSettingsStore.getState();

        const first = buildEvolutionRunConfig(settings);
        const second = buildEvolutionRunConfig(settings);

        expect(first).toEqual(second);
        expect(first.secondary_objectives).toEqual(['latency', 'model_size']);
        expect(first.stopping_policy.policy_type).toBe('all');
        expect(first.device_constraints.mops_budget).toBe(1600);
    });

    it('throws when multi-objective has no secondary goals', () => {
        useEvolutionSettingsStore.setState({
            mobjEnabled: true,
            secondaryObjectives: [],
        });

        expect(() => buildEvolutionRunConfig(useEvolutionSettingsStore.getState())).toThrowError();
    });
});
