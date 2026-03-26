import { render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GenerationSnapshot, PopulatedGenome } from '../../entities/genome';
import type { DeviceProfile, GenomeGenealogy, TrainingProfiler } from '../../shared/lib';
import { evaluateGenomeFeasibility } from '../../features/evolution-manager/model/deviceConstraints';
import { EvolutionStudioPage } from './EvolutionStudioPage';

const useEvolutionLoopMock = vi.fn();
const dashboardPropsSpy = vi.fn();

const settingsStoreMock: any = {
    setGenerationProfilingStat: vi.fn(),
    setGenealogyTree: vi.fn(),
    stoppingPolicy: {
        criteria: [
            { type: 'TargetAccuracy', threshold: 0.9 },
            { type: 'GenerationLimit', max_generations: 100 },
        ],
        policy_type: 'any',
    },
    currentStoppingProgress: {
        generationsSoFar: 3,
        elapsedSeconds: 25,
        plateauPatience: 0,
        bestAccuracySoFar: 0.92,
        triggeredCriteria: ['TargetAccuracy reached'],
    },
    customDeviceParams: undefined,
    selectedDeviceProfile: {
        device_id: 'edge-1',
        device_name: 'Edge Test Device',
        compute_capability: 'ARM',
        ram_mb: 32,
        inference_latency_budget_ms: 10,
        training_available: false,
        max_model_size_mb: 5,
    } as DeviceProfile,
    resourceTargets: {
        flash: 16 * 1024 * 1024,
        ram: 8 * 1024 * 1024,
        macs: 200_000_000,
    },
    paretoHistory: new Map(),
    setParetoForGeneration: vi.fn((generation: number, front: any) => {
        settingsStoreMock.paretoHistory.set(generation, front);
        settingsStoreMock.currentParetoFront = front.pareto_members;
    }),
    showOnlyFeasible: false,
    currentParetoFront: [],
    genealogyTree: undefined,
    useMaxGenerations: false,
    maxGenerations: 100,
    useRandomInitialization: false,
};

const datasetStoreState: {
    selectedProfileId: string | undefined;
    profiles: Array<{ id: string; name: string; type: string; streams: Array<{ shape: number[] }> }>;
} = {
    selectedProfileId: 'dataset-1',
    profiles: [
        {
            id: 'dataset-1',
            name: 'Dataset 1',
            type: 'image',
            streams: [{ shape: [28, 28, 1] }, { shape: [10] }],
        },
    ],
};

const canvasStoreState = {
    genomes: new Map(),
};

const genomeLibraryStoreState = {
    entries: [],
    loadGenomeContent: vi.fn(),
};

vi.mock('../../widgets/title-bar/TitleBar', () => ({
    TitleBar: () => <div>TitleBar</div>,
}));

vi.mock('../../features/evolution-studio', () => ({
    useEvolutionLoop: (...args: unknown[]) => useEvolutionLoopMock(...args),
}));

vi.mock('../../features/evolution-manager', async () => {
    return {
        useEvolutionSettingsStore: () => settingsStoreMock,
        evaluateGenomeFeasibility,
        StoppingCriteriaLiveMonitor: () => <div>MockStoppingCriteriaLiveMonitor</div>,
        StoppingCriteriaSummary: () => <div>MockStoppingCriteriaSummary</div>,
    };
});

vi.mock('../../features/dataset-manager', () => ({
    useDatasetManagerStore: (selector: (state: typeof datasetStoreState) => unknown) => selector(datasetStoreState),
}));

vi.mock('../../entities/canvas-genome', () => ({
    useCanvasGenomeStore: (selector: (state: typeof canvasStoreState) => unknown) => selector(canvasStoreState),
    serializeGenome: vi.fn(),
}));

vi.mock('../../features/genome-library', () => ({
    useGenomeLibraryStore: () => genomeLibraryStoreState,
    GenomeCatalogPicker: () => <div>MockGenomeCatalogPicker</div>,
}));

vi.mock('../../features/genome-library/ui/GenomeDetailPanel', () => ({
    GenomeDetailPanel: () => <div>MockGenomeDetailPanel</div>,
}));

vi.mock('../../shared/hooks', () => ({
    useProfilerStats: () => ({
        avgTrainingTime: 800,
        avgInferenceLatency: 8,
        avgThroughput: 120,
    }),
}));

vi.mock('./EvolutionSettingsPanel', () => ({
    EvolutionSettingsPanel: () => <div>MockEvolutionSettingsPanel</div>,
}));

vi.mock('../../entities/canvas-genome/ui/GenomeSvgPreview/GenomeSvgPreview', () => ({
    GenomeSvgPreview: () => <div>MockGenomeSvgPreview</div>,
}));

vi.mock('./InspectGenomeModal', () => ({
    InspectGenomeModal: () => <div>MockInspectGenomeModal</div>,
}));

vi.mock('../../features/evolution-studio/ui/GenomeProfilerModal', () => ({
    GenomeProfilerModal: () => <div>MockGenomeProfilerModal</div>,
}));

vi.mock('../../features/evolution-studio/ui/ExportGenomeWithWeightsModal', () => ({
    ExportGenomeWithWeightsModal: () => <div>MockExportGenomeWithWeightsModal</div>,
}));

vi.mock('../../widgets/evolution-dashboard', () => ({
    EvolutionDashboard: (props: unknown) => {
        dashboardPropsSpy(props);
        return <div>MockEvolutionDashboard</div>;
    },
}));

vi.mock('../../widgets/post-evolution-panel', () => ({
    PostEvolutionPanel: () => <div>MockPostEvolutionPanel</div>,
}));

function buildLoopState(overrides?: Partial<ReturnType<typeof useEvolutionLoopMock>>) {
    return {
        isRunning: false,
        isPaused: false,
        startEvolution: vi.fn(),
        stopEvolution: vi.fn(),
        pauseEvolution: vi.fn(),
        resumeEvolution: vi.fn(),
        saveCheckpoint: vi.fn(),
        generation: 3,
        population: [],
        hallOfFame: [],
        logs: [],
        currentEvaluatingIndex: 0,
        liveMetrics: [],
        generationHistory: [],
        ...overrides,
    };
}

function profiler(totalTrainMs: number, inferenceMs: number): TrainingProfiler {
    return {
        train_start_ms: 0,
        first_batch_ms: 1,
        train_end_ms: totalTrainMs,
        total_train_duration_ms: totalTrainMs,
        val_start_ms: totalTrainMs,
        val_end_ms: totalTrainMs + 10,
        val_duration_ms: 10,
        test_start_ms: totalTrainMs + 10,
        test_end_ms: totalTrainMs + 20,
        test_duration_ms: 10,
        peak_active_memory_mb: 32,
        peak_model_params_mb: 8,
        peak_gradient_mb: 8,
        peak_optim_state_mb: 8,
        peak_activation_mb: 8,
        samples_per_sec: 100,
        inference_msec_per_sample: inferenceMs,
        batch_count: 4,
    };
}

function makeGenome(
    id: string,
    accuracy: number,
    flashMb: number,
    inferenceMs: number,
): PopulatedGenome {
    return {
        id,
        genome: {} as any,
        nodes: [],
        accuracy,
        adjustedFitness: accuracy,
        resources: {
            totalFlash: flashMb * 1024 * 1024,
            totalRam: 0,
            totalMacs: 0,
            totalNodes: 1,
        },
        profiler: profiler(800, inferenceMs),
    };
}

describe('EvolutionStudioPage integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        settingsStoreMock.paretoHistory = new Map([
            [
                3,
                {
                    generation: 3,
                    total_genomes: 2,
                    pareto_members: [
                        {
                            genome_id: 'g-feasible',
                            accuracy: 0.92,
                            inference_latency_ms: 5,
                            model_size_mb: 1,
                            training_time_ms: 800,
                            is_dominated: false,
                            domination_count: 0,
                        },
                    ],
                    objectives_3d: [[0.92, 5, 1]],
                    all_genomes: [
                        {
                            genome_id: 'g-feasible',
                            accuracy: 0.92,
                            inference_latency_ms: 5,
                            model_size_mb: 1,
                            training_time_ms: 800,
                            is_dominated: false,
                            domination_count: 0,
                        },
                        {
                            genome_id: 'g-slow',
                            accuracy: 0.8,
                            inference_latency_ms: 40,
                            model_size_mb: 4,
                            training_time_ms: 800,
                            is_dominated: true,
                            domination_count: 1,
                        },
                    ],
                    frontier_genome_ids: ['g-feasible'],
                },
            ],
        ]);
        settingsStoreMock.currentParetoFront = [
            {
                genome_id: 'g-feasible',
                accuracy: 0.92,
                inference_latency_ms: 5,
                model_size_mb: 1,
                training_time_ms: 800,
                is_dominated: false,
                domination_count: 0,
            },
        ];
        settingsStoreMock.setGenerationProfilingStat.mockClear();
        settingsStoreMock.setGenealogyTree.mockClear();
        settingsStoreMock.setParetoForGeneration.mockClear();

        const genealogy = new Map<string, GenomeGenealogy>([
            [
                'g-offspring',
                {
                    genome_id: 'g-offspring',
                    generation: 3,
                    parent_ids: ['g-parent-a', 'g-parent-b'],
                    mutation_type: { type: 'Crossover', data: { parent1: 'g-parent-a', parent2: 'g-parent-b' } },
                    mutation_params: {},
                    fitness: 0.92,
                    accuracy: 0.92,
                    created_at_ms: 1000,
                },
            ],
        ]);

        const snapshot: GenerationSnapshot = {
            generation: 3,
            genomes: [
                makeGenome('g-feasible', 0.92, 1, 5),
                makeGenome('g-slow', 0.80, 4, 40),
            ],
            bestFitness: 0.92,
            avgNodes: 2,
            timestamp: '12:00:00',
            evaluated: true,
            totalTrainingMs: 1600,
            totalInferenceMs: 120,
            avgSamplesPerSec: 50,
            genealogy,
        };

        useEvolutionLoopMock.mockReturnValue(
            buildLoopState({
                generation: 3,
                population: snapshot.genomes,
                hallOfFame: [snapshot.genomes[0]],
                generationHistory: [snapshot],
            }),
        );
    });

    it('propagates pareto, device feasibility, stopping reason and profiler stats across page->store->dashboard', async () => {
        render(<EvolutionStudioPage />);

        await waitFor(() => {
            expect(settingsStoreMock.setParetoForGeneration).toHaveBeenCalledWith(
                3,
                expect.objectContaining({ total_genomes: 2 }),
            );
            expect(settingsStoreMock.setGenerationProfilingStat).toHaveBeenCalledWith(
                3,
                expect.objectContaining({ totalTrainingMs: 1600 }),
            );
            expect(settingsStoreMock.setGenealogyTree).toHaveBeenCalledWith(expect.any(Map));
        });

        const latestCallIndex = dashboardPropsSpy.mock.calls.length - 1;
        const latestDashboardProps = dashboardPropsSpy.mock.calls[latestCallIndex]?.[0] as {
            feasibilityByGenomeId: Record<string, boolean>;
            triggeredCriterionIndex: number | null;
            generationHistory: GenerationSnapshot[];
        };

        expect(latestDashboardProps.feasibilityByGenomeId['g-feasible']).toBe(true);
        expect(latestDashboardProps.feasibilityByGenomeId['g-slow']).toBe(false);
        expect(latestDashboardProps.triggeredCriterionIndex).toBe(0);
        expect(
            latestDashboardProps.generationHistory[0].genomes[0].profiler?.total_train_duration_ms,
        ).toBe(800);
    });
});
