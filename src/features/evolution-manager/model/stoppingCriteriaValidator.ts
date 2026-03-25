import type { StoppingCriterionType } from '../../../shared/lib';

/**
 * Validates a single stopping criterion
 */
export function validateSingleCriterion(criterion: StoppingCriterionType): string | null {
    switch (criterion.type) {
        case 'GenerationLimit':
            if (criterion.max_generations <= 0) {
                return 'Max generations must be greater than 0';
            }
            return null;

        case 'FitnessPlateau':
            if (criterion.patience_generations <= 0) {
                return 'Patience must be greater than 0';
            }
            if (criterion.improvement_threshold < 0) {
                return 'Improvement threshold cannot be negative';
            }
            if (!['best_fitness', 'pareto_coverage', 'population_avg'].includes(criterion.monitor)) {
                return 'Invalid monitor value';
            }
            return null;

        case 'TimeLimit':
            if (criterion.max_seconds <= 0) {
                return 'Time limit must be greater than 0 seconds';
            }
            return null;

        case 'TargetAccuracy':
            if (criterion.threshold < 0 || criterion.threshold > 1) {
                return 'Target accuracy must be between 0 and 1';
            }
            return null;

        case 'ManualStop':
            return null;

        default:
            return 'Unknown criterion type';
    }
}

/**
 * Validates the entire stopping policy configuration
 */
export function validateStoppingCriteria(
    criteria: StoppingCriterionType[],
    policyType: 'any' | 'all',
): string | null {
    if (criteria.length === 0) {
        return 'At least one stopping criterion is required';
    }

    // Check for duplicate ManualStop
    const manualStops = criteria.filter(c => c.type === 'ManualStop');
    if (manualStops.length > 1) {
        return 'Only one ManualStop criterion is allowed';
    }

    // Validate each criterion
    for (const criterion of criteria) {
        const error = validateSingleCriterion(criterion);
        if (error) {
            return error;
        }
    }

    // Validate policy type
    if (!['any', 'all'].includes(policyType)) {
        return 'Invalid policy type';
    }

    return null;
}

/**
 * Check if criteria configuration is valid synchronously
 */
export function isStoppingPolicyValid(
    criteria: StoppingCriterionType[],
    policyType: 'any' | 'all',
): boolean {
    return validateStoppingCriteria(criteria, policyType) === null;
}

/**
 * Get a human-readable description of a criterion
 */
export function getCriterionDescription(criterion: StoppingCriterionType): string {
    switch (criterion.type) {
        case 'GenerationLimit':
            return `Stop after ${criterion.max_generations} generations`;

        case 'FitnessPlateau':
            const monitorLabel = {
                best_fitness: 'best fitness',
                pareto_coverage: 'pareto coverage',
                population_avg: 'population average',
            }[criterion.monitor];
            return `Stop if ${monitorLabel} plateaus (patience: ${criterion.patience_generations} gen, threshold: ${criterion.improvement_threshold.toFixed(6)})`;

        case 'TimeLimit':
            const hours = Math.floor(criterion.max_seconds / 3600);
            const minutes = Math.floor((criterion.max_seconds % 3600) / 60);
            const seconds = criterion.max_seconds % 60;
            let timeStr = '';
            if (hours > 0) timeStr += `${hours}h `;
            if (minutes > 0) timeStr += `${minutes}min `;
            if (seconds > 0 || timeStr === '') timeStr += `${seconds}s`;
            return `Stop after ${timeStr.trim()}`;

        case 'TargetAccuracy':
            return `Stop when accuracy reaches ${(criterion.threshold * 100).toFixed(1)}%`;

        case 'ManualStop':
            return 'Can be stopped manually';

        default:
            return 'Unknown criterion';
    }
}
