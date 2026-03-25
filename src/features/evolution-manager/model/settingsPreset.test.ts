import { describe, expect, it } from 'vitest';
import {
    applySettingsPreset,
    createSettingsPreset,
    loadLastUsedSettingsFromLocalStorage,
    loadSettingsPresetFromLocalStorage,
    saveLastUsedSettingsToLocalStorage,
    saveSettingsPresetToLocalStorage,
} from './settingsPreset';
import { useEvolutionSettingsStore } from './store';

describe('settingsPreset', () => {
    it('saves and restores preset snapshots via localStorage', () => {
        const settings = useEvolutionSettingsStore.getState();
        settings.setMobjEnabled(true);
        settings.setSecondaryObjectives(['latency', 'model_size']);
        settings.setMemoryMode('runtime');

        const preset = createSettingsPreset(useEvolutionSettingsStore.getState());
        saveSettingsPresetToLocalStorage(preset);
        saveLastUsedSettingsToLocalStorage(preset);

        const loadedPreset = loadSettingsPresetFromLocalStorage();
        const loadedLastUsed = loadLastUsedSettingsFromLocalStorage();

        expect(loadedPreset).toBeTruthy();
        expect(loadedLastUsed).toBeTruthy();

        useEvolutionSettingsStore.setState({
            mobjEnabled: false,
            secondaryObjectives: [],
            memoryMode: 'estimate',
        });

        applySettingsPreset(useEvolutionSettingsStore.getState(), loadedPreset!);
        expect(useEvolutionSettingsStore.getState().mobjEnabled).toBe(true);
        expect(useEvolutionSettingsStore.getState().secondaryObjectives).toEqual(['latency', 'model_size']);
        expect(useEvolutionSettingsStore.getState().memoryMode).toBe('runtime');
    });
});
