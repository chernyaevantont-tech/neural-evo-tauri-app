import type { GenomeObjectives } from '../../../shared/lib';

export interface DeviceConstraintParams {
    mops_budget: number;
    ram_mb: number;
    flash_mb: number;
    latency_budget_ms: number;
}

export interface DeviceConstraintValidation {
    fieldErrors: Partial<Record<keyof DeviceConstraintParams, string>>;
    warnings: string[];
    isValid: boolean;
}

export interface DeviceFeasibilityResult {
    isFeasible: boolean;
    violationScore: number;
}

function safeRatio(usage: number, budget: number): number {
    if (budget <= 0) {
        return Number.POSITIVE_INFINITY;
    }
    return usage / budget;
}

function estimateMops(objective: GenomeObjectives): number {
    return Math.max(0, objective.inference_latency_ms) * 1.5 + Math.max(0, objective.model_size_mb) * 12.0;
}

function estimateRamMb(objective: GenomeObjectives): number {
    return Math.max(0, objective.model_size_mb) * 2.5;
}

function estimateFlashMb(objective: GenomeObjectives): number {
    return Math.max(0, objective.model_size_mb);
}

export function validateDeviceConstraintParams(
    params: Partial<DeviceConstraintParams>,
): DeviceConstraintValidation {
    const fieldErrors: Partial<Record<keyof DeviceConstraintParams, string>> = {};
    const warnings: string[] = [];

    const entries: Array<keyof DeviceConstraintParams> = [
        'mops_budget',
        'ram_mb',
        'flash_mb',
        'latency_budget_ms',
    ];

    for (const key of entries) {
        const value = params[key];
        if (typeof value !== 'number' || Number.isNaN(value)) {
            fieldErrors[key] = 'Enter a valid number';
            continue;
        }
        if (value <= 0) {
            fieldErrors[key] = 'Value must be greater than 0';
        }
    }

    if ((params.mops_budget ?? 0) > 0 && (params.mops_budget ?? 0) < 10) {
        warnings.push('MOPS budget is very low and may reject nearly all genomes.');
    }
    if ((params.ram_mb ?? 0) > 0 && (params.ram_mb ?? 0) < 4) {
        warnings.push('RAM budget below 4 MB is extremely restrictive.');
    }
    if ((params.flash_mb ?? 0) > 0 && (params.flash_mb ?? 0) < 2) {
        warnings.push('FLASH budget below 2 MB is extremely restrictive.');
    }
    if ((params.latency_budget_ms ?? 0) > 0 && (params.latency_budget_ms ?? 0) < 3) {
        warnings.push('Latency budget below 3 ms may be unrealistic for most devices.');
    }

    return {
        fieldErrors,
        warnings,
        isValid: Object.keys(fieldErrors).length === 0,
    };
}

export function evaluateGenomeFeasibility(
    objective: GenomeObjectives,
    constraints: DeviceConstraintParams,
): DeviceFeasibilityResult {
    const mopsRatio = safeRatio(estimateMops(objective), constraints.mops_budget);
    const ramRatio = safeRatio(estimateRamMb(objective), constraints.ram_mb);
    const flashRatio = safeRatio(estimateFlashMb(objective), constraints.flash_mb);
    const latencyRatio = safeRatio(Math.max(0, objective.inference_latency_ms), constraints.latency_budget_ms);

    const mopsExcess = Math.max(0, mopsRatio - 1);
    const ramExcess = Math.max(0, ramRatio - 1);
    const flashExcess = Math.max(0, flashRatio - 1);
    const latencyExcess = Math.max(0, latencyRatio - 1);

    const violationScore =
        latencyExcess * latencyExcess * 0.35 +
        mopsExcess * mopsExcess * 0.3 +
        ramExcess * ramExcess * 0.2 +
        flashExcess * flashExcess * 0.15;

    const isFeasible = mopsRatio <= 1 && ramRatio <= 1 && flashRatio <= 1 && latencyRatio <= 1;

    return { isFeasible, violationScore };
}
