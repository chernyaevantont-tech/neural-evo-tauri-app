import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useEvolutionSettingsStore } from '../../model/store';
import { ObjectivesSection } from './ObjectivesSection';

function resetObjectiveState() {
    useEvolutionSettingsStore.setState({
        mobjEnabled: false,
        secondaryObjectives: ['latency', 'model_size'],
        objectiveWeightsEnabled: true,
        objectiveWeights: {
            accuracy: 0.5,
            latency: 0.2,
            model_size: 0.2,
            train_time: 0.1,
        },
    });
}

describe('ObjectivesSection', () => {
    beforeEach(() => {
        resetObjectiveState();
    });

    it('switches between single and multi-objective modes', () => {
        render(<ObjectivesSection />);

        fireEvent.change(screen.getByDisplayValue('Single-Objective'), {
            target: { value: 'multi' },
        });

        expect(useEvolutionSettingsStore.getState().mobjEnabled).toBe(true);
    });

    it('shows validation when multi-objective has no secondary goals', () => {
        useEvolutionSettingsStore.setState({
            mobjEnabled: true,
            secondaryObjectives: [],
        });

        render(<ObjectivesSection />);
        expect(screen.getByText('Multi-objective mode requires at least one secondary objective.')).toBeTruthy();
    });

    it('normalizes objective weights on demand', () => {
        useEvolutionSettingsStore.setState({
            mobjEnabled: true,
            objectiveWeights: {
                accuracy: 3,
                latency: 1,
                model_size: 1,
                train_time: 0,
            },
        });

        render(<ObjectivesSection />);
        fireEvent.click(screen.getByRole('button', { name: 'Normalize weights' }));

        const weights = useEvolutionSettingsStore.getState().objectiveWeights;
        const sum = weights.accuracy + weights.latency + weights.model_size + weights.train_time;
        expect(sum).toBeCloseTo(1, 5);
    });
});
