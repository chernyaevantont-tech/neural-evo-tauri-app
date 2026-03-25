import { invoke } from '@tauri-apps/api/core';
import type { GenomeGenealogy, MutationType } from './dtos';

export interface GenomeLineageRecordDto {
    genome_id: string;
    generation: number;
    parent_ids: string[];
    mutation_type: MutationType;
    created_at_unix_ms: number;
}

export interface GenealogyPathDto {
    target_genome_id: string;
    records: GenomeLineageRecordDto[];
    edges: [string, string][];
}

export async function getGenealogyPath(genomeId: string): Promise<GenealogyPathDto> {
    return invoke<GenealogyPathDto>('get_genealogy', { genomeId });
}

export async function getAncestors(genomeId: string, depth?: number): Promise<GenomeLineageRecordDto[]> {
    return invoke<GenomeLineageRecordDto[]>('get_ancestors', {
        genomeId,
        depth: depth === undefined ? null : depth,
    });
}

export async function getDescendants(genomeId: string, depth?: number): Promise<GenomeLineageRecordDto[]> {
    return invoke<GenomeLineageRecordDto[]>('get_descendants', {
        genomeId,
        depth: depth === undefined ? null : depth,
    });
}

export function lineageRecordToGenomeGenealogy(record: GenomeLineageRecordDto): GenomeGenealogy {
    return {
        genome_id: record.genome_id,
        generation: record.generation,
        parent_ids: record.parent_ids,
        mutation_type: record.mutation_type,
        mutation_params: {},
        fitness: 0,
        accuracy: 0,
        created_at_ms: record.created_at_unix_ms,
    };
}
