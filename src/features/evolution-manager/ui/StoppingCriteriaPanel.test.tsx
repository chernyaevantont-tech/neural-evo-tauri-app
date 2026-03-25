import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { StoppingCriteriaPanel } from './StoppingCriteriaPanel';
import { useEvolutionSettingsStore } from '../model/store';

function resetStoppingState() {
    useEvolutionSettingsStore.setState({
        stoppingPolicy: {
            criteria: [{ type: 'ManualStop' }],
            policy_type: 'any',
        },
    });
}

describe('StoppingCriteriaPanel', () => {
    beforeEach(() => {
        resetStoppingState();
    });

    it('adds and removes criteria', () => {
        render(<StoppingCriteriaPanel />);

        fireEvent.click(screen.getByRole('button', { name: '+ Add Criterion' }));
        fireEvent.click(screen.getByRole('button', { name: 'Time Limit' }));

        expect(useEvolutionSettingsStore.getState().stoppingPolicy.criteria).toHaveLength(2);
        expect(screen.getByText('Time Limit')).toBeTruthy();

        const removeButtons = screen.getAllByRole('button', { name: '✕' });
        fireEvent.click(removeButtons[1]);

        expect(useEvolutionSettingsStore.getState().stoppingPolicy.criteria).toHaveLength(1);
    });

    it('renders parameter controls by criterion type', () => {
        useEvolutionSettingsStore.setState({
            stoppingPolicy: {
                criteria: [
                    { type: 'GenerationLimit', max_generations: 200 },
                    {
                        type: 'FitnessPlateau',
                        patience_generations: 12,
                        improvement_threshold: 0.002,
                        monitor: 'population_avg',
                    },
                ],
                policy_type: 'any',
            },
        });

        render(<StoppingCriteriaPanel />);

        expect(screen.getByDisplayValue('200')).toBeTruthy();
        const selectedMonitorOption = screen.getByRole('option', { name: 'Population Avg' });
        expect((selectedMonitorOption as HTMLOptionElement).selected).toBe(true);
        expect(screen.getByDisplayValue('12')).toBeTruthy();
    });

    it('prevents adding a second ManualStop', () => {
        render(<StoppingCriteriaPanel />);

        fireEvent.click(screen.getByRole('button', { name: '+ Add Criterion' }));

        const manualStopOption = screen.getByRole('button', {
            name: 'Manual Stop (Already added)',
        });
        expect((manualStopOption as HTMLButtonElement).disabled).toBe(true);
    });
});
