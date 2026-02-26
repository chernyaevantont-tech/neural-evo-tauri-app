import { create } from 'zustand';

export type CrossoverStrategy = 'subgraph-insertion' | 'subgraph-replacement' | 'neat-style' | 'multi-point';

export interface EvolutionSettingsState {
    // Crossover
    selectedCrossovers: CrossoverStrategy[];
    toggleCrossover: (strategy: CrossoverStrategy) => void;

    // Mutation Probabilities
    mutationRates: {
        params: number;
        addNode: number;
        removeNode: number;
        addSkipConnection: number;
        changeLayerType: number;
    };
    setMutationRate: (key: keyof EvolutionSettingsState['mutationRates'], value: number) => void;

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
        addSkipConnection: 0.3,
        changeLayerType: 0.1,
    },
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
}));
