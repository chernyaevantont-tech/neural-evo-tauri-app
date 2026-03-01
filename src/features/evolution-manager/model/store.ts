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
