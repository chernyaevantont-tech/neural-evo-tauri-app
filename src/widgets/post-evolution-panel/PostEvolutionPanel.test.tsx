import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { PostEvolutionPanel } from './PostEvolutionPanel';

vi.mock('../genealogy-tree-viewer', () => ({
    GenealogicTreeView: () => <div data-testid="genealogy-view">genealogy-view</div>,
}));

vi.mock('../pareto-front-visualizer', () => ({
    ParetoFrontVisualizer: () => <div data-testid="pareto-view">pareto-view</div>,
}));

vi.mock('../../features/genome-library', () => ({
    useGenomeLibraryStore: (selector: (state: { listHiddenLibrary: () => Promise<Array<{ id: string }>> }) => unknown) =>
        selector({
            listHiddenLibrary: async () => [{ id: 'h1' }, { id: 'h2' }],
        }),
}));

describe('PostEvolutionPanel', () => {
    it('renders feasible/infeasible legend and comparison flow', async () => {
        const onExportWeights = vi.fn();

        render(
            <MemoryRouter>
                <PostEvolutionPanel
                    paretoHistory={new Map([
                        [1, {
                            generation: 1,
                            total_genomes: 2,
                            pareto_members: [
                                {
                                    genome_id: 'g1',
                                    accuracy: 0.91,
                                    inference_latency_ms: 5,
                                    model_size_mb: 1.1,
                                    training_time_ms: 100,
                                    is_dominated: false,
                                    domination_count: 0,
                                    device_feasible: true,
                                },
                            ],
                            objectives_3d: [[0.91, 5, 1.1]],
                            all_genomes: [
                                {
                                    genome_id: 'g1',
                                    accuracy: 0.91,
                                    inference_latency_ms: 5,
                                    model_size_mb: 1.1,
                                    training_time_ms: 100,
                                    is_dominated: false,
                                    domination_count: 0,
                                    device_feasible: true,
                                },
                                {
                                    genome_id: 'g2',
                                    accuracy: 0.88,
                                    inference_latency_ms: 7,
                                    model_size_mb: 1.4,
                                    training_time_ms: 140,
                                    is_dominated: true,
                                    domination_count: 1,
                                    device_feasible: false,
                                },
                            ],
                            frontier_genome_ids: ['g1'],
                        }],
                    ])}
                    generation={1}
                    elapsedRuntimeSeconds={12}
                    stoppingPolicy={[{ type: 'ManualStop' }]}
                    stoppingReason="manual"
                    genomeById={new Map()}
                    feasibilityByGenomeId={{ g1: true, g2: false }}
                    onExportWeights={onExportWeights}
                />
            </MemoryRouter>,
        );

        expect(screen.getByTestId('legend-feasible')).toBeTruthy();
        expect(screen.getByTestId('legend-infeasible')).toBeTruthy();

        const compareSelect = screen.getByTestId('comparison-select');
        fireEvent.change(compareSelect, { target: { value: 'g2' } });

        await waitFor(() => {
            const selected = screen.getByText(/Selected:/);
            expect(selected.textContent).toContain('g1');
        });

        fireEvent.click(screen.getByRole('button', { name: 'Select and Export Model Weights' }));
        expect(onExportWeights).toHaveBeenCalledWith('g1');
    });

    it('invokes continue evolution callback', async () => {
        const onContinueEvolution = vi.fn();

        render(
            <MemoryRouter>
                <PostEvolutionPanel
                    paretoHistory={new Map()}
                    generation={0}
                    elapsedRuntimeSeconds={0}
                    stoppingPolicy={[{ type: 'ManualStop' }]}
                    stoppingReason=""
                    genomeById={new Map()}
                    onContinueEvolution={onContinueEvolution}
                />
            </MemoryRouter>,
        );

        await waitFor(() => {
            expect(screen.getByText(/N genomes auto-saved:/).textContent).toContain('2');
        });

        fireEvent.click(screen.getByRole('button', { name: 'Continue Evolution' }));
        expect(onContinueEvolution).toHaveBeenCalledTimes(1);
    });
});
