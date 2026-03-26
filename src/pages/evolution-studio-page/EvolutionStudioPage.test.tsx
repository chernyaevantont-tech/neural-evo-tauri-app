import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EvolutionStudioPage } from './EvolutionStudioPage';

const useEvolutionLoopMock = vi.fn();

const settingsStoreMock: any = {
    setGenerationProfilingStat: vi.fn(),
    setGenealogyTree: vi.fn(),
    stoppingPolicy: { criteria: [{ type: 'ManualStop' }], policy_type: 'any' },
    currentStoppingProgress: { generationsSoFar: 0, elapsedSeconds: 0, plateauPatience: 0, bestAccuracySoFar: 0 },
    customDeviceParams: undefined,
    selectedDeviceProfile: undefined,
    resourceTargets: { flash: 1024 * 1024, ram: 256 * 1024, macs: 1_000_000 },
    paretoHistory: new Map(),
    setParetoForGeneration: vi.fn(),
    showOnlyFeasible: false,
    currentParetoFront: [],
    genealogyTree: undefined,
    useMaxGenerations: false,
    maxGenerations: 100,
    useRandomInitialization: false,
};

const datasetStoreState: {
    selectedProfileId: string | undefined;
    profiles: Array<{ id: string; name: string; type: string }>;
} = {
    selectedProfileId: undefined,
    profiles: [],
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

vi.mock('../../features/evolution-manager', () => ({
    useEvolutionSettingsStore: () => settingsStoreMock,
    evaluateGenomeFeasibility: () => ({ isFeasible: true, violationScore: 0, violated: [] }),
    StoppingCriteriaLiveMonitor: () => <div>MockStoppingCriteriaLiveMonitor</div>,
    StoppingCriteriaSummary: () => <div>MockStoppingCriteriaSummary</div>,
}));

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
        avgTrainingTime: 0,
        avgInferenceLatency: 0,
        avgThroughput: 0,
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
    EvolutionDashboard: () => <div>MockEvolutionDashboard</div>,
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
        generation: 0,
        population: [],
        hallOfFame: [],
        logs: [],
        currentEvaluatingIndex: 0,
        liveMetrics: [],
        generationHistory: [],
        ...overrides,
    };
}

describe('EvolutionStudioPage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        datasetStoreState.selectedProfileId = undefined;
        datasetStoreState.profiles = [];
        useEvolutionLoopMock.mockReturnValue(buildLoopState());
    });

    it('renders page layout and composes settings with dashboard', () => {
        render(<EvolutionStudioPage />);

        expect(screen.getByText('Evolution Studio')).toBeTruthy();
        expect(screen.getByText('MockEvolutionSettingsPanel')).toBeTruthy();
        expect(screen.getByText('MockEvolutionDashboard')).toBeTruthy();
        expect(screen.getByText('No dataset selected. Go to Dataset Manager to configure one.')).toBeTruthy();
    });

    it('shows live stopping monitor while evolution is running', () => {
        useEvolutionLoopMock.mockReturnValue(
            buildLoopState({
                isRunning: true,
                population: [{ id: 'g1', nodes: [], genome: {} }],
            }),
        );

        render(<EvolutionStudioPage />);

        expect(screen.getByText('MockStoppingCriteriaLiveMonitor')).toBeTruthy();
        expect(screen.queryByText('MockStoppingCriteriaSummary')).toBeNull();
    });

    it('shows completion summary and post-run panel after finished generation history', async () => {
        useEvolutionLoopMock.mockReturnValue(
            buildLoopState({
                generation: 5,
                generationHistory: [
                    {
                        generation: 5,
                        evaluated: true,
                        genomes: [],
                    },
                ],
            }),
        );

        render(<EvolutionStudioPage />);

        await waitFor(() => {
            expect(screen.getByText('MockStoppingCriteriaSummary')).toBeTruthy();
            expect(screen.getByText('MockPostEvolutionPanel')).toBeTruthy();
        });
    });
});