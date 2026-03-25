import { describe, expect, it } from 'vitest';
import {
    buildFitnessTimeline,
    buildJobs,
    deriveOverviewMetrics,
    formatDuration,
    formatEta,
    formatMemory,
} from './dashboardSelectors';
import type { GenerationSnapshot } from '../../../entities/genome';
import type { GenerationParetoFront } from '../../../shared/lib';

describe('dashboardSelectors', () => {
    it('formats duration and eta', () => {
        expect(formatDuration(3661)).toBe('01:01:01');
        expect(formatEta(undefined)).toBe('data unavailable');
        expect(formatEta(9)).toBe('00:00:09');
    });

    it('formats memory', () => {
        expect(formatMemory(512)).toBe('512.0 MB');
        expect(formatMemory(2048)).toBe('2.00 GB');
        expect(formatMemory(undefined)).toBe('data unavailable');
    });

    it('derives overview metrics with eta and feasible ratio values', () => {
        const snapshot: GenerationSnapshot = {
            generation: 1,
            genomes: [{ id: 'g1', genome: {} as any, nodes: [], adjustedFitness: 0.8 }],
            bestFitness: 0.8,
            avgNodes: 5,
            timestamp: '12:00:00',
            evaluated: true,
        };

        const result = deriveOverviewMetrics({
            generation: 1,
            generationHistory: [snapshot],
            elapsedTimeSeconds: 20,
            useMaxGenerations: true,
            maxGenerations: 5,
            currentParetoFrontSize: 2,
            feasibleByGenomeId: { g1: true, g2: false },
        });

        expect(result.genomesEvaluated).toBe(1);
        expect(result.currentBestFitness).toBe(0.8);
        expect(result.etaSeconds).toBe(80);
        expect(result.feasibleSolutions).toBe(1);
        expect(result.feasibleTotal).toBe(2);
    });

    it('builds timeline with feasible front and constraint pressure', () => {
        const generationHistory: GenerationSnapshot[] = [
            {
                generation: 2,
                genomes: [
                    { id: 'g1', genome: {} as any, nodes: [], adjustedFitness: 0.8 },
                    { id: 'g2', genome: {} as any, nodes: [], adjustedFitness: 0.6 },
                ],
                bestFitness: 0.8,
                avgNodes: 4,
                timestamp: '12:00:00',
                evaluated: true,
            },
        ];

        const pareto: GenerationParetoFront = {
            generation: 2,
            total_genomes: 2,
            pareto_members: [
                {
                    genome_id: 'g1',
                    accuracy: 0.9,
                    inference_latency_ms: 10,
                    model_size_mb: 2,
                    training_time_ms: 100,
                    is_dominated: false,
                    domination_count: 0,
                },
                {
                    genome_id: 'g2',
                    accuracy: 0.8,
                    inference_latency_ms: 12,
                    model_size_mb: 2.2,
                    training_time_ms: 100,
                    is_dominated: false,
                    domination_count: 0,
                },
            ],
            objectives_3d: [],
            all_genomes: [],
            frontier_genome_ids: ['g1', 'g2'],
        };

        const timeline = buildFitnessTimeline(
            generationHistory,
            new Map([[2, pareto]]),
            { g1: true, g2: false },
            { g1: 0.1, g2: 0.5 },
        );

        expect(timeline).toHaveLength(1);
        expect(timeline[0].avgFitness).toBeCloseTo(0.7, 5);
        expect(timeline[0].feasibleFrontSize).toBe(1);
        expect(timeline[0].constraintPressure).toBeCloseTo(0.3, 5);
    });

    it('builds jobs with running status and progress', () => {
        const jobs = buildJobs({
            isRunning: true,
            currentEvaluatingIndex: 1,
            liveMetrics: [{ epoch: 1, batch: 5, total_batches: 10, loss: 0.3, accuracy: 70 }],
            population: [
                { id: 'g1', genome: {} as any, nodes: [] },
                { id: 'g2', genome: {} as any, nodes: [] },
            ],
        });

        expect(jobs[0].status).toBe('completed');
        expect(jobs[1].status).toBe('running');
        expect(jobs[1].progressPercent).toBe(50);
    });
});
