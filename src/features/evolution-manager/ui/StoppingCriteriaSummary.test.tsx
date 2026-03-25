import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { StoppingCriteriaSummary } from './StoppingCriteriaSummary';

describe('StoppingCriteriaSummary', () => {
    it('renders stop reason and final statistics', () => {
        render(
            <StoppingCriteriaSummary
                triggeredCriterionIndex={1}
                criteria={[
                    { type: 'GenerationLimit', max_generations: 100 },
                    { type: 'TargetAccuracy', threshold: 0.95 },
                ]}
                finalGeneration={37}
                elapsedSeconds={3725}
                finalAccuracy={0.953}
            />,
        );

        expect(screen.getByText('Evolution Stopped')).toBeTruthy();
        expect(screen.getByText('Triggered by')).toBeTruthy();
        expect(screen.getByText('TargetAccuracy')).toBeTruthy();
        expect(screen.getByText('Final Generation')).toBeTruthy();
        expect(screen.getByText('37')).toBeTruthy();
        expect(screen.getByText('01:02:05')).toBeTruthy();
        expect(screen.getByText('95.30%')).toBeTruthy();
    });
});
