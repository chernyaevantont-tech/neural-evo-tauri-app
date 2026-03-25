import { describe, expect, it } from 'vitest';
import type { GenomeGenealogy, GenomeObjectives } from '../../../shared/lib';
import {
    buildComparisonRows,
    buildEvolutionReportDataModel,
    buildLineageExport,
    buildParetoExportPayload,
} from './reportBuilders';

describe('reportBuilders', () => {
    it('builds comparison rows with device ratios and lineage depth', () => {
        const objectives = new Map<string, GenomeObjectives>([
            ['g1', {
                genome_id: 'g1',
                accuracy: 0.93,
                inference_latency_ms: 4,
                model_size_mb: 1.5,
                training_time_ms: 100,
                is_dominated: false,
                domination_count: 0,
            }],
        ]);

        const genealogy = new Map<string, GenomeGenealogy>([
            ['p1', {
                genome_id: 'p1',
                generation: 0,
                parent_ids: [],
                mutation_type: { type: 'Random' },
                mutation_params: {},
                fitness: 0.4,
                accuracy: 0.5,
                created_at_ms: 1,
            }],
            ['g1', {
                genome_id: 'g1',
                generation: 1,
                parent_ids: ['p1'],
                mutation_type: { type: 'AddNode', data: { node_type: 'Dense', source: 'a', target: 'b' } },
                mutation_params: {},
                fitness: 0.8,
                accuracy: 0.9,
                created_at_ms: 2,
            }],
        ]);

        const rows = buildComparisonRows({
            selectedGenomeIds: ['g1'],
            objectivesByGenomeId: objectives,
            genomeById: new Map(),
            genealogyTree: genealogy,
            activeDeviceConstraints: {
                mops_budget: 10,
                ram_mb: 32,
                flash_mb: 8,
                latency_budget_ms: 20,
            },
        });

        expect(rows).toHaveLength(1);
        expect(rows[0].lineageDepth).toBe(1);
        expect(rows[0].deviceRatios.flash).toBeCloseTo(1.5 / 8, 6);
    });

    it('builds pareto export payload json', () => {
        const json = buildParetoExportPayload({
            generation: 3,
            total_genomes: 2,
            pareto_members: [],
            objectives_3d: [],
            all_genomes: [],
            frontier_genome_ids: ['a', 'b'],
        });

        const parsed = JSON.parse(json) as { generation: number; frontier_genome_ids: string[] };
        expect(parsed.generation).toBe(3);
        expect(parsed.frontier_genome_ids).toEqual(['a', 'b']);
    });

    it('builds report data model with constraints and stopping reason', () => {
        const model = buildEvolutionReportDataModel({
            generation: 7,
            elapsedRuntimeSeconds: 44,
            stoppingPolicy: [{ type: 'ManualStop' }],
            stoppingReason: 'manual',
            paretoFront: {
                generation: 7,
                total_genomes: 1,
                pareto_members: [{
                    genome_id: 'g7',
                    accuracy: 0.88,
                    inference_latency_ms: 6,
                    model_size_mb: 1.3,
                    training_time_ms: 120,
                    is_dominated: false,
                    domination_count: 0,
                }],
                objectives_3d: [[0.88, 6, 1.3]],
                all_genomes: [{
                    genome_id: 'g7',
                    accuracy: 0.88,
                    inference_latency_ms: 6,
                    model_size_mb: 1.3,
                    training_time_ms: 120,
                    is_dominated: false,
                    domination_count: 0,
                }],
            },
            constraints: {
                mops_budget: 12,
                ram_mb: 48,
                flash_mb: 16,
                latency_budget_ms: 25,
            },
            hiddenArchive: {
                total: 3,
                avgFitness: 0.7,
                minAccuracy: 0.5,
                maxAccuracy: 0.9,
                feasibleCount: 2,
            },
        });

        expect(model.runConfig.generation).toBe(7);
        expect(model.constraintsSummary?.mops_budget).toBe(12);
        expect(model.stoppingReason).toBe('manual');
    });

    it('exports lineage as graphml', () => {
        const tree = new Map<string, GenomeGenealogy>([
            ['a', {
                genome_id: 'a',
                generation: 0,
                parent_ids: [],
                mutation_type: { type: 'Random' },
                mutation_params: {},
                fitness: 0.2,
                accuracy: 0.3,
                created_at_ms: 1,
            }],
            ['b', {
                genome_id: 'b',
                generation: 1,
                parent_ids: ['a'],
                mutation_type: { type: 'RemoveNode', data: { node_id: 'x' } },
                mutation_params: {},
                fitness: 0.4,
                accuracy: 0.5,
                created_at_ms: 2,
            }],
        ]);

        const xml = buildLineageExport(tree, 'graphml');
        expect(xml).toContain('<node id="a"/>');
        expect(xml).toContain('<edge source="a" target="b"/>');
    });
});
