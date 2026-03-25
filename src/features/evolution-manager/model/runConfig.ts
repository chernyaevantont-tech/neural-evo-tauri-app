import type { StoppingPolicy } from '../../../shared/lib';
import type { EvolutionSettingsState, ObjectiveWeightKey, SecondaryObjective } from './store';
import {
    normalizeObjectiveWeights,
    normalizeSecondaryObjectives,
    validateObjectives,
} from './objectives';

export interface EvolutionRunConfig {
    objective_mode: 'single' | 'multi';
    primary_objective: 'accuracy';
    secondary_objectives: SecondaryObjective[];
    objective_weights_enabled: boolean;
    objective_weights: Record<ObjectiveWeightKey, number>;
    device_constraints: {
        mops_budget: number;
        ram_mb: number;
        flash_mb: number;
        latency_budget_ms: number;
    };
    stopping_policy: StoppingPolicy;
    advanced_performance: {
        profiling_enabled: boolean;
        safety_margin_mb: number;
        estimator_safety_factor: number;
        memory_mode: 'estimate' | 'runtime' | 'hybrid';
    };
}

function coerceDeviceConstraints(settings: EvolutionSettingsState): EvolutionRunConfig['device_constraints'] {
    const custom = settings.customDeviceParams;
    if (custom) {
        return {
            mops_budget: custom.mops_budget ?? Math.max(1, settings.resourceTargets.macs / 1_000_000),
            ram_mb: custom.ram_mb,
            flash_mb: custom.flash_mb ?? custom.max_model_size_mb ?? Math.max(1, settings.resourceTargets.flash / (1024 * 1024)),
            latency_budget_ms: custom.latency_budget_ms,
        };
    }

    return {
        mops_budget: Math.max(1, settings.resourceTargets.macs / 1_000_000),
        ram_mb: settings.selectedDeviceProfile?.ram_mb ?? Math.max(1, settings.resourceTargets.ram / (1024 * 1024)),
        flash_mb:
            settings.selectedDeviceProfile?.max_model_size_mb ??
            Math.max(1, settings.resourceTargets.flash / (1024 * 1024)),
        latency_budget_ms: settings.selectedDeviceProfile?.inference_latency_budget_ms ?? 100,
    };
}

export function buildEvolutionRunConfig(settings: EvolutionSettingsState): EvolutionRunConfig {
    const objectiveMode = settings.mobjEnabled ? 'multi' : 'single';
    const secondary = normalizeSecondaryObjectives(settings.secondaryObjectives);
    const objectiveError = validateObjectives(objectiveMode, secondary);
    if (objectiveError) {
        throw new Error(objectiveError);
    }

    const weights = normalizeObjectiveWeights(settings.objectiveWeights, secondary);

    return {
        objective_mode: objectiveMode,
        primary_objective: settings.primaryObjective,
        secondary_objectives: secondary,
        objective_weights_enabled: settings.objectiveWeightsEnabled,
        objective_weights: weights,
        device_constraints: coerceDeviceConstraints(settings),
        stopping_policy: settings.stoppingPolicy,
        advanced_performance: {
            profiling_enabled: settings.profilingEnabled,
            safety_margin_mb: settings.memorySafetyMarginMb,
            estimator_safety_factor: settings.estimatorSafetyFactor,
            memory_mode: settings.memoryMode,
        },
    };
}
