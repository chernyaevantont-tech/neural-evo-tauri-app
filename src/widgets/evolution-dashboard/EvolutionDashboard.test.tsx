import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EvolutionDashboard } from './EvolutionDashboard';
import type { GenerationSnapshot } from '../../entities/genome';

vi.mock('react-chartjs-2', () => ({
    Line: () => <div data-testid="line-chart" />,
}));

const paretoHistory = new Map([
    [
        1,
        {
            generation: 1,
            total_genomes: 1,
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
            ],
            objectives_3d: [[0.9, 10, 2]],
            all_genomes: [],
            frontier_genome_ids: ['g1'],
        },
    ],
]);

const generationHistory: GenerationSnapshot[] = [
    {
        generation: 1,
        genomes: [
            {
                id: 'g1',
                genome: {} as any,
                nodes: [],
                adjustedFitness: 0.95,
                accuracy: 90,
                profiler: {
                    train_start_ms: 0,
                    total_train_duration_ms: 1200,
                    first_batch_ms: 200,
                    train_end_ms: 1200,
                    val_start_ms: 1200,
                    val_end_ms: 1500,
                    batch_count: 8,
                    samples_per_sec: 120,
                    val_duration_ms: 300,
                    test_start_ms: 1500,
                    test_end_ms: 1610,
                    inference_msec_per_sample: 1.23,
                    test_duration_ms: 110,
                    peak_active_memory_mb: 64,
                    peak_model_params_mb: 10,
                    peak_gradient_mb: 20,
                    peak_optim_state_mb: 12,
                    peak_activation_mb: 22,
                },
            },
        ],
        bestFitness: 0.95,
        avgNodes: 4,
        timestamp: '12:00:00',
        evaluated: true,
    },
];

describe('EvolutionDashboard', () => {
    beforeEach(() => {
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('renders overview cards and jobs panel', () => {
        render(
            <EvolutionDashboard
                isRunning={true}
                isPaused={false}
                generation={1}
                generationHistory={generationHistory}
                liveMetrics={[{ epoch: 1, batch: 2, total_batches: 4, loss: 0.4, accuracy: 70 }]}
                currentEvaluatingIndex={0}
                population={generationHistory[0].genomes}
                logs={[{ time: '12:00:00', message: 'hello', type: 'info' }]}
                elapsedRuntimeSeconds={30}
                useMaxGenerations={true}
                maxGenerations={10}
                currentParetoFront={paretoHistory.get(1)!.pareto_members}
                paretoHistory={paretoHistory as any}
                feasibilityByGenomeId={{ g1: true }}
                constraintViolationScoreByGenomeId={{ g1: 0.2 }}
                showOnlyFeasible={false}
                genealogyTree={new Map()}
                onUseAsSeed={() => {}}
                onOpenGenomeDetails={() => {}}
                onExportSelected={() => {}}
                stoppingCriteria={[]}
                triggeredCriterionIndex={null}
                bestAccuracyNormalized={0.9}
                onPause={() => {}}
                onResume={() => {}}
                onStop={() => {}}
                onSaveCheckpoint={() => {}}
                onOpenProfiler={() => {}}
            />,
        );

        expect(screen.getByText('Generations elapsed')).toBeTruthy();
        expect(screen.getByText('Active jobs')).toBeTruthy();
        expect(screen.getByRole('button', { name: 'running' })).toBeTruthy();
    });

    it('switches tabs and calls control handlers', () => {
        const onPause = vi.fn();
        const onResume = vi.fn();
        const onStop = vi.fn();
        const onSaveCheckpoint = vi.fn();

        render(
            <EvolutionDashboard
                isRunning={true}
                isPaused={false}
                generation={1}
                generationHistory={generationHistory}
                liveMetrics={[]}
                currentEvaluatingIndex={0}
                population={generationHistory[0].genomes}
                logs={[]}
                elapsedRuntimeSeconds={15}
                useMaxGenerations={true}
                maxGenerations={5}
                currentParetoFront={paretoHistory.get(1)!.pareto_members}
                paretoHistory={paretoHistory as any}
                feasibilityByGenomeId={{ g1: true }}
                constraintViolationScoreByGenomeId={{ g1: 0 }}
                showOnlyFeasible={false}
                genealogyTree={new Map()}
                onUseAsSeed={() => {}}
                onOpenGenomeDetails={() => {}}
                onExportSelected={() => {}}
                stoppingCriteria={[]}
                triggeredCriterionIndex={null}
                bestAccuracyNormalized={0.9}
                onPause={onPause}
                onResume={onResume}
                onStop={onStop}
                onSaveCheckpoint={onSaveCheckpoint}
                onOpenProfiler={() => {}}
            />,
        );

        fireEvent.click(screen.getByRole('button', { name: 'Genealogy Tree' }));
        fireEvent.click(screen.getByRole('button', { name: 'Performance Metrics' }));

        fireEvent.click(screen.getByRole('button', { name: 'Pause Evolution' }));
        fireEvent.click(screen.getByRole('button', { name: 'Stop Evolution' }));
        fireEvent.click(screen.getByRole('button', { name: 'Save Checkpoint' }));

        expect(onPause).toHaveBeenCalled();
        expect(onStop).toHaveBeenCalled();
        expect(onSaveCheckpoint).toHaveBeenCalled();
        expect(onResume).not.toHaveBeenCalled();
    });
});
