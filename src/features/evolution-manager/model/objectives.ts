import type { ObjectiveWeightKey, SecondaryObjective } from './store';

export type ObjectiveMode = 'single' | 'multi';

const SECONDARY_ORDER: SecondaryObjective[] = ['latency', 'model_size', 'train_time', 'training_time', 'energy'];

export function normalizeSecondaryObjectives(values: SecondaryObjective[]): SecondaryObjective[] {
    const unique = Array.from(new Set(values));
    return unique.sort((a, b) => SECONDARY_ORDER.indexOf(a) - SECONDARY_ORDER.indexOf(b));
}

export function canonicalObjectiveKey(value: SecondaryObjective): ObjectiveWeightKey | null {
    if (value === 'latency' || value === 'model_size') {
        return value;
    }
    if (value === 'train_time' || value === 'training_time') {
        return 'train_time';
    }
    return null;
}

export function validateObjectives(mode: ObjectiveMode, secondaryObjectives: SecondaryObjective[]): string | null {
    if (mode === 'single') {
        return null;
    }

    const normalized = normalizeSecondaryObjectives(secondaryObjectives);
    // In multi-objective mode, primary accuracy + at least one secondary objective => minimum 2 goals.
    if (normalized.length < 1) {
        return 'Multi-objective mode requires at least one secondary objective.';
    }

    return null;
}

export function normalizeObjectiveWeights(
    weights: Record<ObjectiveWeightKey, number>,
    activeSecondary: SecondaryObjective[],
): Record<ObjectiveWeightKey, number> {
    const activeKeys = new Set<ObjectiveWeightKey>(['accuracy']);

    for (const objective of activeSecondary) {
        const key = canonicalObjectiveKey(objective);
        if (key) {
            activeKeys.add(key);
        }
    }

    const keys = Array.from(activeKeys);
    const sum = keys.reduce((acc, key) => acc + Math.max(0, weights[key] ?? 0), 0);
    if (sum <= Number.EPSILON) {
        const uniform = 1 / keys.length;
        return {
            accuracy: activeKeys.has('accuracy') ? uniform : 0,
            latency: activeKeys.has('latency') ? uniform : 0,
            model_size: activeKeys.has('model_size') ? uniform : 0,
            train_time: activeKeys.has('train_time') ? uniform : 0,
        };
    }

    return {
        accuracy: activeKeys.has('accuracy') ? Math.max(0, weights.accuracy) / sum : 0,
        latency: activeKeys.has('latency') ? Math.max(0, weights.latency) / sum : 0,
        model_size: activeKeys.has('model_size') ? Math.max(0, weights.model_size) / sum : 0,
        train_time: activeKeys.has('train_time') ? Math.max(0, weights.train_time) / sum : 0,
    };
}
