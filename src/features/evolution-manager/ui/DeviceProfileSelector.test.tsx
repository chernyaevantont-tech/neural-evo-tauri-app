import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DeviceProfileSelector } from './DeviceProfileSelector';
import { useEvolutionSettingsStore } from '../model/store';

function resetDeviceState() {
    useEvolutionSettingsStore.setState({
        deviceProfileId: 'embedded-mcu',
        isCustomDevice: false,
        showOnlyFeasible: false,
        selectedDeviceProfile: undefined,
        customDeviceParams: {
            mops_budget: 120,
            ram_mb: 0.5,
            flash_mb: 2,
            latency_budget_ms: 80,
            max_model_size_mb: 2,
        },
        resourceTargets: {
            flash: 2 * 1024 * 1024,
            ram: Math.round(0.5 * 1024 * 1024),
            macs: 120_000_000,
        },
    });
}

describe('DeviceProfileSelector', () => {
    beforeEach(() => {
        resetDeviceState();
    });

    it('switches to custom constraints mode and edits values', () => {
        render(<DeviceProfileSelector />);

        const toggle = screen.getByLabelText('Custom constraints');
        fireEvent.click(toggle);
        expect(useEvolutionSettingsStore.getState().isCustomDevice).toBe(true);

        const mopsInput = screen.getByLabelText('MOPS budget');
        fireEvent.change(mopsInput, { target: { value: '1500' } });

        expect(useEvolutionSettingsStore.getState().customDeviceParams?.mops_budget).toBe(1500);
        expect(useEvolutionSettingsStore.getState().resourceTargets.macs).toBe(1_500_000_000);
    });

    it('updates show only feasible filter in store', () => {
        render(<DeviceProfileSelector />);

        const checkbox = screen.getByLabelText('Show only feasible');
        fireEvent.click(checkbox);

        expect(useEvolutionSettingsStore.getState().showOnlyFeasible).toBe(true);
    });

    it('calls template extension-point callbacks', () => {
        const onSaveAsTemplate = vi.fn();
        const onLoadTemplate = vi.fn();

        render(
            <DeviceProfileSelector
                onSaveAsTemplate={onSaveAsTemplate}
                onLoadTemplate={onLoadTemplate}
            />,
        );

        fireEvent.change(screen.getByLabelText('Template name'), { target: { value: 'Edge Custom' } });
        fireEvent.click(screen.getByRole('button', { name: 'Save' }));
        fireEvent.click(screen.getByRole('button', { name: 'Load template' }));

        expect(onSaveAsTemplate).toHaveBeenCalledWith('Edge Custom');
        expect(onLoadTemplate).toHaveBeenCalledWith(useEvolutionSettingsStore.getState().deviceProfileId);
    });
});
