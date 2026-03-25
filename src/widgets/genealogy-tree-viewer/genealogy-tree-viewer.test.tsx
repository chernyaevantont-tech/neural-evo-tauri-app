import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { GenerationParetoFront, GenomeGenealogy } from '../../shared/lib';
import { GenealogicTreeView } from './GenealogicTreeView';

function makeGenome(
    genomeId: string,
    generation: number,
    parentIds: string[],
    fitness: number,
): GenomeGenealogy {
    return {
        genome_id: genomeId,
        generation,
        parent_ids: parentIds,
        mutation_type: parentIds.length === 0 ? { type: 'Random' } : { type: 'AddNode', data: { node_type: 'Dense', source: 'a', target: 'b' } },
        mutation_params: {},
        fitness,
        accuracy: fitness,
        created_at_ms: generation * 1000,
    };
}

describe('GenealogicTreeView', () => {
    const tree = new Map<string, GenomeGenealogy>([
        ['g1', makeGenome('g1', 0, [], 0.5)],
        ['g2', makeGenome('g2', 1, ['g1'], 0.7)],
        ['g3', makeGenome('g3', 2, ['g2'], 0.9)],
    ]);

    const paretoHistory = new Map<number, GenerationParetoFront>([
        [
            2,
            {
                generation: 2,
                total_genomes: 3,
                pareto_members: [
                    {
                        genome_id: 'g2',
                        accuracy: 0.7,
                        inference_latency_ms: 10,
                        model_size_mb: 1,
                        training_time_ms: 100,
                        is_dominated: false,
                        domination_count: 0,
                    },
                ],
                objectives_3d: [[0.7, 10, 1]],
                all_genomes: [
                    {
                        genome_id: 'g1',
                        accuracy: 0.5,
                        inference_latency_ms: 8,
                        model_size_mb: 0.9,
                        training_time_ms: 90,
                        is_dominated: false,
                        domination_count: 0,
                    },
                    {
                        genome_id: 'g2',
                        accuracy: 0.7,
                        inference_latency_ms: 10,
                        model_size_mb: 1,
                        training_time_ms: 100,
                        is_dominated: false,
                        domination_count: 0,
                    },
                    {
                        genome_id: 'g3',
                        accuracy: 0.9,
                        inference_latency_ms: 12,
                        model_size_mb: 1.1,
                        training_time_ms: 120,
                        is_dominated: false,
                        domination_count: 0,
                    },
                ],
                frontier_genome_ids: ['g2'],
            },
        ],
    ]);

    it('opens selected node details on click', () => {
        const onOpen = vi.fn();

        render(
            <GenealogicTreeView
                genealogyTree={tree}
                paretoHistory={paretoHistory}
                onOpenGenomeDetails={onOpen}
            />,
        );

        fireEvent.click(screen.getByTestId('genealogy-node-g2'));
        expect(screen.getByText('Genome g2')).toBeTruthy();

        fireEvent.click(screen.getByRole('button', { name: 'Open details' }));
        expect(onOpen).toHaveBeenCalledWith('g2');
    });

    it('updates rendering when generation filter changes', () => {
        render(<GenealogicTreeView genealogyTree={tree} paretoHistory={paretoHistory} />);

        expect(screen.getByTestId('genealogy-node-g1')).toBeTruthy();

        fireEvent.change(screen.getByLabelText('Generation min'), { target: { value: '2' } });

        expect(screen.queryByTestId('genealogy-node-g1')).toBeNull();
        expect(screen.getByTestId('genealogy-node-g3')).toBeTruthy();
    });
});
