import { beforeEach, describe, expect, it } from 'vitest';
import { useEvolutionSettingsStore } from './store';

describe('useEvolutionSettingsStore device actions', () => {
    beforeEach(() => {
        useEvolutionSettingsStore.setState({
            showOnlyFeasible: false,
            isCustomDevice: false,
            deviceProfileId: 'default-device',
            customDeviceParams: undefined,
            selectedDeviceProfile: undefined,
            resourceTargets: {
                flash: 1024 * 1024,
                ram: 256 * 1024,
                macs: 1_000_000,
            },
        });
    });

    it('toggles showOnlyFeasible with dedicated action', () => {
        useEvolutionSettingsStore.getState().setShowOnlyFeasible(true);
        expect(useEvolutionSettingsStore.getState().showOnlyFeasible).toBe(true);
    });

    it('accepts extended custom device params payload', () => {
        useEvolutionSettingsStore.getState().setCustomDeviceParams({
            mops_budget: 1200,
            ram_mb: 64,
            flash_mb: 128,
            latency_budget_ms: 20,
            max_model_size_mb: 128,
        });

        expect(useEvolutionSettingsStore.getState().customDeviceParams).toEqual({
            mops_budget: 1200,
            ram_mb: 64,
            flash_mb: 128,
            latency_budget_ms: 20,
            max_model_size_mb: 128,
        });
    });
});
