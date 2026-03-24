import { useCallback } from 'react';
import type { GenomeGenealogy } from '../lib/dtos';

export function useGenealogy() {
    const buildAncestralChain = useCallback(
        (genomeId: string, genealogyMap: Map<string, GenomeGenealogy>): string[] => {
            const chain = [genomeId];
            let current = genealogyMap.get(genomeId);

            while (current && current.parent_ids.length > 0) {
                const parent = current.parent_ids[0];
                chain.push(parent);
                current = genealogyMap.get(parent);
            }

            return chain;
        },
        [],
    );

    const hasCycles = useCallback((genealogyMap: Map<string, GenomeGenealogy>): boolean => {
        for (const [genomeId] of genealogyMap) {
            const visited = new Set<string>();
            let current: string | undefined = genomeId;

            while (current && !visited.has(current)) {
                visited.add(current);
                const node = genealogyMap.get(current);
                if (!node || node.parent_ids.length === 0) {
                    current = undefined;
                } else {
                    current = node.parent_ids[0];
                }
            }

            if (current && visited.has(current)) {
                return true;
            }
        }

        return false;
    }, []);

    return { buildAncestralChain, hasCycles };
}
