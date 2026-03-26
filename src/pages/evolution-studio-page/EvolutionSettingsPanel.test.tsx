import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EvolutionSettingsPanel } from './EvolutionSettingsPanel';

const buildEvolutionRunConfigMock = vi.fn();
const createSettingsPresetMock = vi.fn(() => ({ preset: true }));
const saveLastUsedSettingsToLocalStorageMock = vi.fn();
const saveSettingsPresetToLocalStorageMock = vi.fn();
const loadLastUsedSettingsFromLocalStorageMock = vi.fn();
const applySettingsPresetMock = vi.fn();

const toggleCrossoverMock = vi.fn();
const setMutationRateMock = vi.fn();
const setUseAdaptiveMutationMock = vi.fn();
const setAdaptiveTargetNodesMock = vi.fn();
const setUseMaxNodesLimitMock = vi.fn();
const setMaxNodesLimitMock = vi.fn();
const setUseParsimonyPressureMock = vi.fn();
const setParsimonyAlphaMock = vi.fn();
const setUseResourceAwareFitnessMock = vi.fn();
const setResourceTargetMock = vi.fn();
const setBatchSizeMock = vi.fn();
const setEvalEpochsMock = vi.fn();
const setDatasetPercentMock = vi.fn();
const setPopulationSizeMock = vi.fn();
const setUseMaxGenerationsMock = vi.fn();
const setMaxGenerationsMock = vi.fn();
const setUseRandomInitializationMock = vi.fn();
const setRandomInitRatioMock = vi.fn();
const setUseZeroCostProxiesMock = vi.fn();
const setZeroCostStrategyMock = vi.fn();
const setFastPassThresholdMock = vi.fn();
const setPartialTrainingEpochsMock = vi.fn();
const normalizeObjectiveWeightsMock = vi.fn();

let settingsMock: any;

vi.mock('../../features/evolution-manager', () => ({
    useEvolutionSettingsStore: () => settingsMock,
    buildEvolutionRunConfig: buildEvolutionRunConfigMock,
    createSettingsPreset: createSettingsPresetMock,
    saveLastUsedSettingsToLocalStorage: saveLastUsedSettingsToLocalStorageMock,
    saveSettingsPresetToLocalStorage: saveSettingsPresetToLocalStorageMock,
    loadLastUsedSettingsFromLocalStorage: loadLastUsedSettingsFromLocalStorageMock,
    applySettingsPreset: applySettingsPresetMock,
    ObjectivesSection: () => <div>ObjectivesSection</div>,
    DeviceTargetingSection: () => <div>DeviceTargetingSection</div>,
    StoppingCriteriaSection: () => <div>StoppingCriteriaSection</div>,
    AdvancedPerformanceSection: () => <div>AdvancedPerformanceSection</div>,
}));

function resetSettingsMock() {
    settingsMock = {
        selectedCrossovers: ['subgraph-insertion'],
        toggleCrossover: toggleCrossoverMock,
        mutationRates: {
            params: 0.6,
            addNode: 0.2,
            removeNode: 0.1,
            removeSubgraph: 0.05,
            addSkipConnection: 0.3,
            changeLayerType: 0.1,
        },
        setMutationRate: setMutationRateMock,
        useAdaptiveMutation: false,
        setUseAdaptiveMutation: setUseAdaptiveMutationMock,
        adaptiveTargetNodes: 20,
        setAdaptiveTargetNodes: setAdaptiveTargetNodesMock,
        useMaxNodesLimit: false,
        maxNodesLimit: 30,
        setUseMaxNodesLimit: setUseMaxNodesLimitMock,
        setMaxNodesLimit: setMaxNodesLimitMock,
        useParsimonyPressure: false,
        parsimonyAlpha: 0.01,
        setUseParsimonyPressure: setUseParsimonyPressureMock,
        setParsimonyAlpha: setParsimonyAlphaMock,
        useResourceAwareFitness: false,
        setUseResourceAwareFitness: setUseResourceAwareFitnessMock,
        resourceTargets: {
            flash: 1024,
            ram: 2048,
            macs: 4096,
        },
        setResourceTarget: setResourceTargetMock,
        useRandomInitialization: false,
        setUseRandomInitialization: setUseRandomInitializationMock,
        randomInitRatio: 30,
        setRandomInitRatio: setRandomInitRatioMock,
        batchSize: 32,
        setBatchSize: setBatchSizeMock,
        evalEpochs: 1,
        setEvalEpochs: setEvalEpochsMock,
        datasetPercent: 100,
        setDatasetPercent: setDatasetPercentMock,
        populationSize: 20,
        setPopulationSize: setPopulationSizeMock,
        useMaxGenerations: false,
        setUseMaxGenerations: setUseMaxGenerationsMock,
        maxGenerations: 100,
        setMaxGenerations: setMaxGenerationsMock,
        useZeroCostProxies: false,
        setUseZeroCostProxies: setUseZeroCostProxiesMock,
        zeroCostStrategy: 'two-stage',
        setZeroCostStrategy: setZeroCostStrategyMock,
        fastPassThreshold: 0.6,
        setFastPassThreshold: setFastPassThresholdMock,
        partialTrainingEpochs: 20,
        setPartialTrainingEpochs: setPartialTrainingEpochsMock,
        normalizeObjectiveWeights: normalizeObjectiveWeightsMock,
    };
}

describe('EvolutionSettingsPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        resetSettingsMock();
        loadLastUsedSettingsFromLocalStorageMock.mockReturnValue(null);
    });

    it('renders composition sections from features layer', () => {
        render(<EvolutionSettingsPanel />);

        expect(screen.getByText('ObjectivesSection')).toBeTruthy();
        expect(screen.getByText('DeviceTargetingSection')).toBeTruthy();
        expect(screen.getByText('StoppingCriteriaSection')).toBeTruthy();
        expect(screen.getByText('AdvancedPerformanceSection')).toBeTruthy();
    });

    it('applies and validates settings on Apply click', () => {
        render(<EvolutionSettingsPanel />);

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

        expect(normalizeObjectiveWeightsMock).toHaveBeenCalledOnce();
        expect(buildEvolutionRunConfigMock).toHaveBeenCalledWith(settingsMock);
        expect(createSettingsPresetMock).toHaveBeenCalledWith(settingsMock);
        expect(saveLastUsedSettingsToLocalStorageMock).toHaveBeenCalledWith({ preset: true });
        expect(screen.getByText('Settings applied and validated.')).toBeTruthy();
    });

    it('loads last used config and applies preset to store', () => {
        const preset = { profile: 'last-used' };
        loadLastUsedSettingsFromLocalStorageMock.mockReturnValue(preset);

        render(<EvolutionSettingsPanel />);

        fireEvent.click(screen.getByRole('button', { name: 'Load last used config' }));

        expect(applySettingsPresetMock).toHaveBeenCalledWith(settingsMock, preset);
        expect(screen.getByText('Last used config restored.')).toBeTruthy();
    });
});