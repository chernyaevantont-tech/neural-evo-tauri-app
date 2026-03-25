import { create } from 'zustand';
import type {
    DeviceProfile,
    GenerationParetoFront,
    GenomeGenealogy,
    GenomeObjectives,
    StoppingPolicy,
} from '../../../shared/lib';

export type CrossoverStrategy = 'subgraph-insertion' | 'subgraph-replacement' | 'neat-style' | 'multi-point';
export type SecondaryObjective = 'latency' | 'model_size' | 'training_time' | 'energy';

export interface GenerationProfilingStats {
    generation: number;
    totalTrainingMs: number;
    totalInferenceMs: number;
    avgSamplesPerSec: number;
    peakConcurrentVramMb: number;
    totalJobsCompleted: number;
    totalJobsFailed: number;
}

export interface StoppingProgress {
    generationsSoFar: number;
    elapsedSeconds: number;
    plateauPatience: number;
    bestAccuracySoFar: number;
    triggeredCriteria?: string[];
}

export interface EvolutionSettingsState {
    // Crossover
    selectedCrossovers: CrossoverStrategy[];
    toggleCrossover: (strategy: CrossoverStrategy) => void;

    // Mutation Probabilities
    mutationRates: {
        params: number;
        addNode: number;
        removeNode: number;
        removeSubgraph: number;
        addSkipConnection: number;
        changeLayerType: number;
    };
    setMutationRate: (key: keyof EvolutionSettingsState['mutationRates'], value: number) => void;

    // Adaptive Mutations
    useAdaptiveMutation: boolean;
    setUseAdaptiveMutation: (val: boolean) => void;
    adaptiveTargetNodes: number;
    setAdaptiveTargetNodes: (val: number) => void;

    // Bloat Control
    maxNodesLimit: number;
    useMaxNodesLimit: boolean;
    setUseMaxNodesLimit: (val: boolean) => void;
    setMaxNodesLimit: (val: number) => void;

    useParsimonyPressure: boolean;
    parsimonyAlpha: number;
    setUseParsimonyPressure: (val: boolean) => void;
    setParsimonyAlpha: (val: number) => void;

    useResourceAwareFitness: boolean;
    setUseResourceAwareFitness: (val: boolean) => void;

    resourceTargets: {
        flash: number;
        ram: number;
        macs: number;
    };
    setResourceTarget: (key: keyof EvolutionSettingsState['resourceTargets'], value: number) => void;

    // Evaluation Settings
    batchSize: number;
    setBatchSize: (val: number) => void;
    evalEpochs: number;
    setEvalEpochs: (val: number) => void;
    datasetPercent: number;
    setDatasetPercent: (val: number) => void;

    // Population & Generations
    populationSize: number;
    setPopulationSize: (val: number) => void;
    maxGenerations: number;
    useMaxGenerations: boolean;
    setMaxGenerations: (val: number) => void;
    setUseMaxGenerations: (val: boolean) => void;

    // Random Architecture Initialization
    useRandomInitialization: boolean;
    setUseRandomInitialization: (val: boolean) => void;
    randomInitRatio: number; // 0-100: percentage of population to initialize randomly
    setRandomInitRatio: (val: number) => void;

    // Zero-Cost Proxy Evaluation
    useZeroCostProxies: boolean;
    setUseZeroCostProxies: (val: boolean) => void;
    zeroCostStrategy: 'two-stage' | 'early-stopping';
    setZeroCostStrategy: (val: 'two-stage' | 'early-stopping') => void;
    fastPassThreshold: number; // 0.0-1.0: threshold for full training
    setFastPassThreshold: (val: number) => void;
    partialTrainingEpochs: number;
    setPartialTrainingEpochs: (val: number) => void;

    // Performance & Profiling
    profilingEnabled: boolean;
    setProfilingEnabled: (val: boolean) => void;
    memorySafetyMarginMb: number;
    setMemorySafetyMarginMb: (val: number) => void;
    estimatorSafetyFactor: number;
    setEstimatorSafetyFactor: (val: number) => void;

    // Multi-Objective
    mobjEnabled: boolean;
    setMobjEnabled: (val: boolean) => void;
    primaryObjective: 'accuracy';
    secondaryObjectives: SecondaryObjective[];
    setSecondaryObjectives: (val: SecondaryObjective[]) => void;

    // Device Targeting
    deviceProfileId: string;
    setDeviceProfileId: (val: string) => void;
    isCustomDevice: boolean;
    setIsCustomDevice: (val: boolean) => void;
    customDeviceParams?: {
        mops_budget?: number;
        ram_mb: number;
        flash_mb?: number;
        vram_mb?: number;
        latency_budget_ms: number;
        max_model_size_mb?: number;
    };
    setCustomDeviceParams: (val?: EvolutionSettingsState['customDeviceParams']) => void;
    selectedDeviceProfile?: DeviceProfile;
    setSelectedDeviceProfile: (val?: DeviceProfile) => void;
        showOnlyFeasible: boolean;
        setShowOnlyFeasible: (val: boolean) => void;

    // Stopping Criteria
    stoppingPolicy: StoppingPolicy;
    setStoppingPolicy: (policy: StoppingPolicy) => void;

    // Genealogy
    genealogyTrackingEnabled: boolean;
    setGenealogyTrackingEnabled: (val: boolean) => void;

    // Hidden Library
    autoSaveToHiddenLibrary: boolean;
    setAutoSaveToHiddenLibrary: (val: boolean) => void;

    // Pareto & Multi-Objective runtime state
    paretoHistory: Map<number, GenerationParetoFront>;
    currentParetoFront: GenomeObjectives[];
    setCurrentParetoFront: (front: GenomeObjectives[]) => void;
    setParetoForGeneration: (generation: number, front: GenerationParetoFront) => void;

    // Genealogy runtime state
    genealogyTree?: Map<string, GenomeGenealogy>;
    setGenealogyTree: (tree?: Map<string, GenomeGenealogy>) => void;

    // Performance tracking runtime state
    generationProfilingStats: Map<number, GenerationProfilingStats>;
    setGenerationProfilingStat: (generation: number, stats: GenerationProfilingStats) => void;

    // Stopping criteria progress runtime state
    currentStoppingProgress: StoppingProgress;
    setCurrentStoppingProgress: (progress: StoppingProgress) => void;

    // Hidden library auto-save count
    hiddenLibraryGenomeCount: number;
    setHiddenLibraryGenomeCount: (count: number) => void;
    incrementHiddenLibraryGenomeCount: () => void;

    resetAdvancedTracking: () => void;
}

export const useEvolutionSettingsStore = create<EvolutionSettingsState>((set) => ({
    selectedCrossovers: ['subgraph-insertion'],
    toggleCrossover: (strategy) => set((state) => {
        const has = state.selectedCrossovers.includes(strategy);
        if (has && state.selectedCrossovers.length === 1) return state; // Prevent unchecking last
        return {
            selectedCrossovers: has
                ? state.selectedCrossovers.filter(s => s !== strategy)
                : [...state.selectedCrossovers, strategy]
        };
    }),

    mutationRates: {
        params: 0.6,
        addNode: 0.2,
        removeNode: 0.1,
        removeSubgraph: 0.05,
        addSkipConnection: 0.3,
        changeLayerType: 0.1,
    },
    useAdaptiveMutation: false,
    adaptiveTargetNodes: 20,
    setUseAdaptiveMutation: (val) => set({ useAdaptiveMutation: val }),
    setAdaptiveTargetNodes: (val) => set({ adaptiveTargetNodes: val }),
    setMutationRate: (key, value) => set((state) => ({
        mutationRates: { ...state.mutationRates, [key]: value }
    })),

    maxNodesLimit: 30,
    useMaxNodesLimit: false,
    setUseMaxNodesLimit: (val) => set({ useMaxNodesLimit: val }),
    setMaxNodesLimit: (val) => set({ maxNodesLimit: val }),

    useParsimonyPressure: false,
    parsimonyAlpha: 0.01,
    setUseParsimonyPressure: (val) => set({ useParsimonyPressure: val }),
    setParsimonyAlpha: (val) => set({ parsimonyAlpha: val }),

    useResourceAwareFitness: false,
    setUseResourceAwareFitness: (val) => set({ useResourceAwareFitness: val }),

    resourceTargets: {
        flash: 1024 * 1024, // 1MB default
        ram: 256 * 1024,    // 256KB default
        macs: 1000000       // 1M MACs default
    },
    setResourceTarget: (key, value) => set((state) => ({
        resourceTargets: { ...state.resourceTargets, [key]: value }
    })),

    batchSize: 32,
    setBatchSize: (val) => set({ batchSize: val }),
    evalEpochs: 1,
    setEvalEpochs: (val) => set({ evalEpochs: val }),
    datasetPercent: 100,
    setDatasetPercent: (val) => set({ datasetPercent: Math.max(1, Math.min(100, val)) }),

    populationSize: 20,
    setPopulationSize: (val) => set({ populationSize: Math.max(4, Math.min(200, val)) }),
    maxGenerations: 100,
    useMaxGenerations: false,
    setMaxGenerations: (val) => set({ maxGenerations: Math.max(1, val) }),
    setUseMaxGenerations: (val) => set({ useMaxGenerations: val }),

    useRandomInitialization: false,
    setUseRandomInitialization: (val) => set({ useRandomInitialization: val }),
    randomInitRatio: 30, // 30% of population initialized randomly by default
    setRandomInitRatio: (val) => set({ randomInitRatio: Math.max(0, Math.min(100, val)) }),

    useZeroCostProxies: false,
    setUseZeroCostProxies: (val) => set({ useZeroCostProxies: val }),
    zeroCostStrategy: 'two-stage',
    setZeroCostStrategy: (val) => set({ zeroCostStrategy: val }),
    fastPassThreshold: 0.6,
    setFastPassThreshold: (val) => set({ fastPassThreshold: Math.max(0, Math.min(1, val)) }),
    partialTrainingEpochs: 20,
    setPartialTrainingEpochs: (val) => set({ partialTrainingEpochs: Math.max(1, Math.min(100, val)) }),

    profilingEnabled: false,
    setProfilingEnabled: (val) => set({ profilingEnabled: val }),
    memorySafetyMarginMb: 128,
    setMemorySafetyMarginMb: (val) => set({ memorySafetyMarginMb: Math.max(0, val) }),
    estimatorSafetyFactor: 1.1,
    setEstimatorSafetyFactor: (val) => set({ estimatorSafetyFactor: Math.max(1, val) }),

    mobjEnabled: false,
    setMobjEnabled: (val) => set({ mobjEnabled: val }),
    primaryObjective: 'accuracy',
    secondaryObjectives: ['latency', 'model_size'],
    setSecondaryObjectives: (val) => set({ secondaryObjectives: val }),

    deviceProfileId: 'default-device',
    setDeviceProfileId: (val) => set({ deviceProfileId: val }),
    isCustomDevice: false,
    setIsCustomDevice: (val) => set({ isCustomDevice: val }),
    customDeviceParams: undefined,
    setCustomDeviceParams: (val) => set({ customDeviceParams: val }),
    selectedDeviceProfile: undefined,
    setSelectedDeviceProfile: (val) => set({ selectedDeviceProfile: val }),
        showOnlyFeasible: false,
        setShowOnlyFeasible: (val) => set({ showOnlyFeasible: val }),

    stoppingPolicy: {
        criteria: [{ type: 'ManualStop' }],
        policy_type: 'any',
    },
    setStoppingPolicy: (policy) => set({ stoppingPolicy: policy }),

    genealogyTrackingEnabled: true,
    setGenealogyTrackingEnabled: (val) => set({ genealogyTrackingEnabled: val }),

    autoSaveToHiddenLibrary: false,
    setAutoSaveToHiddenLibrary: (val) => set({ autoSaveToHiddenLibrary: val }),

    paretoHistory: new Map(),
    currentParetoFront: [],
    setCurrentParetoFront: (front) => set({ currentParetoFront: front }),
    setParetoForGeneration: (generation, front) =>
        set((state) => {
            const next = new Map(state.paretoHistory);
            next.set(generation, front);
            return {
                paretoHistory: next,
                currentParetoFront: front.pareto_members,
            };
        }),

    genealogyTree: undefined,
    setGenealogyTree: (tree) => set({ genealogyTree: tree }),

    generationProfilingStats: new Map(),
    setGenerationProfilingStat: (generation, stats) =>
        set((state) => {
            const next = new Map(state.generationProfilingStats);
            next.set(generation, stats);
            return { generationProfilingStats: next };
        }),

    currentStoppingProgress: {
        generationsSoFar: 0,
        elapsedSeconds: 0,
        plateauPatience: 0,
        bestAccuracySoFar: 0,
        triggeredCriteria: [],
    },
    setCurrentStoppingProgress: (progress) => set({ currentStoppingProgress: progress }),

    hiddenLibraryGenomeCount: 0,
    setHiddenLibraryGenomeCount: (count) => set({ hiddenLibraryGenomeCount: Math.max(0, count) }),
    incrementHiddenLibraryGenomeCount: () =>
        set((state) => ({ hiddenLibraryGenomeCount: state.hiddenLibraryGenomeCount + 1 })),

    resetAdvancedTracking: () =>
        set({
            paretoHistory: new Map(),
            currentParetoFront: [],
            genealogyTree: undefined,
            generationProfilingStats: new Map(),
            currentStoppingProgress: {
                generationsSoFar: 0,
                elapsedSeconds: 0,
                plateauPatience: 0,
                bestAccuracySoFar: 0,
                triggeredCriteria: [],
            },
            hiddenLibraryGenomeCount: 0,
        }),
}));

export function getAdaptiveMutationRates(currentNodes: number) {
    const state = useEvolutionSettingsStore.getState();
    if (!state.useAdaptiveMutation) {
        return {
            addNode: state.mutationRates.addNode,
            removeNode: state.mutationRates.removeNode,
            removeSubgraph: state.mutationRates.removeSubgraph
        };
    }

    const target = state.adaptiveTargetNodes;
    // If currentNodes < target, increase Add and decrease Remove
    if (currentNodes <= target) {
        const ratio = currentNodes / Math.max(1, target); // 0.0 -> 1.0
        return {
            addNode: Math.max(0.1, 0.4 - 0.2 * ratio),      // High when small, drops near target
            removeNode: Math.max(0.01, 0.05 * ratio),       // Low when small, rises near target
            removeSubgraph: Math.max(0.01, 0.02 * ratio)    // Very low when small
        };
    } else {
        // If currentNodes > target, penalize Add and strictly boost Remove
        const ratio = Math.min(2.0, currentNodes / target); // 1.0 -> 2.0+
        return {
            addNode: Math.max(0.01, 0.2 - 0.1 * ratio),     // Quickly drops to near zero
            removeNode: Math.min(0.8, 0.1 + 0.3 * (ratio - 1)), // Approaches 40-80% as bloat increases
            removeSubgraph: Math.min(0.5, 0.05 + 0.2 * (ratio - 1)) // Rises aggressively to 50% for bloat pruning
        };
    }
}
