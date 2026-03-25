import type { EvolutionSettingsState, MemoryMode, ObjectiveWeightKey, SecondaryObjective } from './store';

const PRESET_STORAGE_KEY = 'evolution-settings-preset-v1';
const LAST_USED_STORAGE_KEY = 'evolution-settings-last-used-v1';

export interface EvolutionSettingsPreset {
    mobjEnabled: boolean;
    secondaryObjectives: SecondaryObjective[];
    objectiveWeightsEnabled: boolean;
    objectiveWeights: Record<ObjectiveWeightKey, number>;
    deviceProfileId: string;
    isCustomDevice: boolean;
    customDeviceParams: EvolutionSettingsState['customDeviceParams'];
    showOnlyFeasible: boolean;
    stoppingPolicy: EvolutionSettingsState['stoppingPolicy'];
    profilingEnabled: boolean;
    memorySafetyMarginMb: number;
    estimatorSafetyFactor: number;
    memoryMode: MemoryMode;
}

function safeParsePreset(raw: string | null): EvolutionSettingsPreset | null {
    if (!raw) {
        return null;
    }

    try {
        return JSON.parse(raw) as EvolutionSettingsPreset;
    } catch {
        return null;
    }
}

export function createSettingsPreset(settings: EvolutionSettingsState): EvolutionSettingsPreset {
    return {
        mobjEnabled: settings.mobjEnabled,
        secondaryObjectives: settings.secondaryObjectives,
        objectiveWeightsEnabled: settings.objectiveWeightsEnabled,
        objectiveWeights: settings.objectiveWeights,
        deviceProfileId: settings.deviceProfileId,
        isCustomDevice: settings.isCustomDevice,
        customDeviceParams: settings.customDeviceParams,
        showOnlyFeasible: settings.showOnlyFeasible,
        stoppingPolicy: settings.stoppingPolicy,
        profilingEnabled: settings.profilingEnabled,
        memorySafetyMarginMb: settings.memorySafetyMarginMb,
        estimatorSafetyFactor: settings.estimatorSafetyFactor,
        memoryMode: settings.memoryMode,
    };
}

export function applySettingsPreset(settings: EvolutionSettingsState, preset: EvolutionSettingsPreset): void {
    settings.setMobjEnabled(preset.mobjEnabled);
    settings.setSecondaryObjectives(preset.secondaryObjectives);
    settings.setObjectiveWeightsEnabled(preset.objectiveWeightsEnabled);
    settings.setObjectiveWeight('accuracy', preset.objectiveWeights.accuracy);
    settings.setObjectiveWeight('latency', preset.objectiveWeights.latency);
    settings.setObjectiveWeight('model_size', preset.objectiveWeights.model_size);
    settings.setObjectiveWeight('train_time', preset.objectiveWeights.train_time);
    settings.setDeviceProfileId(preset.deviceProfileId);
    settings.setIsCustomDevice(preset.isCustomDevice);
    settings.setCustomDeviceParams(preset.customDeviceParams);
    settings.setShowOnlyFeasible(preset.showOnlyFeasible);
    settings.setStoppingPolicy(preset.stoppingPolicy);
    settings.setProfilingEnabled(preset.profilingEnabled);
    settings.setMemorySafetyMarginMb(preset.memorySafetyMarginMb);
    settings.setEstimatorSafetyFactor(preset.estimatorSafetyFactor);
    settings.setMemoryMode(preset.memoryMode);
}

export function saveSettingsPresetToLocalStorage(preset: EvolutionSettingsPreset): void {
    if (typeof window === 'undefined' || !window.localStorage) {
        return;
    }

    window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(preset));
}

export function loadSettingsPresetFromLocalStorage(): EvolutionSettingsPreset | null {
    if (typeof window === 'undefined' || !window.localStorage) {
        return null;
    }

    return safeParsePreset(window.localStorage.getItem(PRESET_STORAGE_KEY));
}

export function saveLastUsedSettingsToLocalStorage(preset: EvolutionSettingsPreset): void {
    if (typeof window === 'undefined' || !window.localStorage) {
        return;
    }

    window.localStorage.setItem(LAST_USED_STORAGE_KEY, JSON.stringify(preset));
}

export function loadLastUsedSettingsFromLocalStorage(): EvolutionSettingsPreset | null {
    if (typeof window === 'undefined' || !window.localStorage) {
        return null;
    }

    return safeParsePreset(window.localStorage.getItem(LAST_USED_STORAGE_KEY));
}
