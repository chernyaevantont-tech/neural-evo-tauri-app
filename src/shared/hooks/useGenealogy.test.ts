import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { GenomeGenealogy } from '../lib/dtos';
import { useGenealogy } from './useGenealogy';

function mockGenome(id: string, parentIds: string[]): GenomeGenealogy {
    return {
        genome_id: id,
        generation: 0,
        parent_ids: parentIds,
        mutation_type: { type: 'Random' },
        mutation_params: {},
        fitness: 0,
        accuracy: 0,
        created_at_ms: Date.now(),
    };
}

describe('useGenealogy', () => {
    it('builds ancestral chain', () => {
        const { result } = renderHook(() => useGenealogy());

        const genealogyMap = new Map<string, GenomeGenealogy>([
            ['g3', mockGenome('g3', ['g2'])],
            ['g2', mockGenome('g2', ['g1'])],
            ['g1', mockGenome('g1', [])],
        ]);

        const chain = result.current.buildAncestralChain('g3', genealogyMap);
        expect(chain).toEqual(['g3', 'g2', 'g1']);
    });

    it('detects cycles in genealogy map', () => {
        const { result } = renderHook(() => useGenealogy());

        const genealogyMap = new Map<string, GenomeGenealogy>([
            ['g1', mockGenome('g1', ['g2'])],
            ['g2', mockGenome('g2', ['g1'])],
        ]);

        expect(result.current.hasCycles(genealogyMap)).toBe(true);
    });
});
