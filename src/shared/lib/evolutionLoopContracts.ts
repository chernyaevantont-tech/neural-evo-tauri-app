export type AdaptiveMutationSettings = {
    useAdaptiveMutation: boolean;
    adaptiveTargetNodes: number;
    mutationRates: {
        addNode: number;
        removeNode: number;
        removeSubgraph: number;
    };
};

export type EvolutionLoopSettings = AdaptiveMutationSettings & {
    populationSize: number;
    useRandomInitialization: boolean;
    randomInitRatio: number;
    useMaxNodesLimit: boolean;
    maxNodesLimit: number;
    mutationRates: {
        params: number;
        addNode: number;
        removeNode: number;
        removeSubgraph: number;
        addSkipConnection: number;
        changeLayerType: number;
    };
    useZeroCostProxies: boolean;
    zeroCostStrategy: 'two-stage' | 'early-stopping';
    fastPassThreshold: number;
    partialTrainingEpochs: number;
    evalEpochs: number;
    batchSize: number;
    datasetPercent: number;
    useParsimonyPressure: boolean;
    parsimonyAlpha: number;
    useResourceAwareFitness: boolean;
    resourceTargets: {
        flash: number;
        ram: number;
        macs: number;
    };
    customDeviceParams?: {
        mops_budget?: number;
        ram_mb: number;
        flash_mb?: number;
        vram_mb?: number;
        latency_budget_ms: number;
        max_model_size_mb?: number;
    };
    memorySafetyMarginMb?: number;
    estimatorSafetyFactor?: number;
    useMaxGenerations: boolean;
    maxGenerations: number;
    genealogyTrackingEnabled: boolean;
    selectedCrossovers: Array<'subgraph-insertion' | 'subgraph-replacement' | 'neat-style' | 'multi-point'>;
};

export type DatasetValidationIssueLite = {
    component: string;
    message: string;
};

export type DatasetValidationReportLite = {
    issues: DatasetValidationIssueLite[];
    input_shapes?: Record<string, number[]>;
    output_shape?: number[];
};

export type DatasetSplit = {
    train: number;
    val: number;
    test: number;
};

export type DatasetProfileLite = {
    id: string;
    streams: Array<{
        role: 'Input' | 'Target' | 'Ignore';
        tensorShape: number[];
        dataType?: 'Image' | 'TemporalSequence' | 'Vector' | 'Categorical' | 'Text';
        numClasses?: number;
    }>;
    split: DatasetSplit;
    totalSamples?: number;
    scanResult?: { totalMatched: number };
    isScanned: boolean;
    isValidForEvolution: boolean;
    validationReport?: DatasetValidationReportLite;
};

export type UseEvolutionLoopParams = {
    datasetProfileId: string | null;
    settings: EvolutionLoopSettings;
    datasetProfiles: DatasetProfileLite[];
};
