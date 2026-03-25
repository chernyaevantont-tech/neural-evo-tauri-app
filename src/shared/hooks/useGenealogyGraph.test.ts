import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { GenomeGenealogy } from '../lib';
import { useGenealogyGraph } from './useGenealogyGraph';

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

describe('useGenealogyGraph', () => {
    const tree = new Map<string, GenomeGenealogy>([
        ['g1', makeGenome('g1', 0, [], 0.5)],
        ['g2', makeGenome('g2', 1, ['g1'], 0.7)],
        ['g3', makeGenome('g3', 2, ['g2'], 0.9)],
    ]);

    it('transforms genealogy map into positioned nodes and edges', () => {
        const { result } = renderHook(() =>
            useGenealogyGraph({
                genealogyTree: tree,
                filters: {
                    generationMin: 0,
                    generationMax: 2,
                    fitnessMin: 0,
                    fitnessMax: 1,
                    paretoOnly: false,
                    ancestorsDepth: 0,
                },
            }),
        );

        expect(result.current.nodes).toHaveLength(3);
        expect(result.current.edges).toHaveLength(2);
        expect(result.current.nodes.every((node) => node.x >= 0 && node.x <= 1)).toBe(true);
        expect(result.current.nodes.every((node) => node.y >= 0 && node.y <= 1)).toBe(true);
    });

    it('applies generation and fitness filters', () => {
        const { result } = renderHook(() =>
            useGenealogyGraph({
                genealogyTree: tree,
                filters: {
                    generationMin: 1,
                    generationMax: 2,
                    fitnessMin: 0.8,
                    fitnessMax: 1,
                    paretoOnly: false,
                    ancestorsDepth: 0,
                },
            }),
        );

        expect(result.current.nodes.map((node) => node.id)).toEqual(['g3']);
        expect(result.current.edges).toHaveLength(0);
    });

    it('keeps ancestors for selected node up to depth even if filtered out', () => {
        const { result } = renderHook(() =>
            useGenealogyGraph({
                genealogyTree: tree,
                filters: {
                    generationMin: 2,
                    generationMax: 2,
                    fitnessMin: 0.9,
                    fitnessMax: 1,
                    paretoOnly: false,
                    ancestorsDepth: 2,
                },
                selectedGenomeId: 'g3',
            }),
        );

        const ids = result.current.nodes.map((node) => node.id).sort();
        expect(ids).toEqual(['g1', 'g2', 'g3']);
    });

    it('supports pareto-only filter', () => {
        const { result } = renderHook(() =>
            useGenealogyGraph({
                genealogyTree: tree,
                filters: {
                    generationMin: 0,
                    generationMax: 2,
                    fitnessMin: 0,
                    fitnessMax: 1,
                    paretoOnly: true,
                    ancestorsDepth: 0,
                },
                paretoGenomeIds: new Set(['g2']),
            }),
        );

        expect(result.current.nodes.map((node) => node.id)).toEqual(['g2']);
    });
});
