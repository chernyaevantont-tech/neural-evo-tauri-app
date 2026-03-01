import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

export interface GenomeLibraryEntry {
    id: string;
    name: string;
    tags: string[];
    createdAt: string;        // ISO timestamp
    inputDims: number[];      // dimensionality of each input, e.g. [3] for one 3D image
    outputDims: number[];     // dimensionality of each output, e.g. [1] for one 1D vector
    totalNodes: number;
    layerTypes: string[];     // unique layer types: ["Conv2D", "Dense", "Pooling"]
    bestLoss?: number;
    bestAccuracy?: number;
}

// Rust response uses snake_case
interface RustGenomeLibraryEntry {
    id: string;
    name: string;
    tags: string[];
    created_at: string;
    input_dims: number[];
    output_dims: number[];
    total_nodes: number;
    layer_types: string[];
    best_loss: number | null;
    best_accuracy: number | null;
}

function mapEntry(r: RustGenomeLibraryEntry): GenomeLibraryEntry {
    return {
        id: r.id,
        name: r.name,
        tags: r.tags,
        createdAt: r.created_at,
        inputDims: r.input_dims,
        outputDims: r.output_dims,
        totalNodes: r.total_nodes,
        layerTypes: r.layer_types,
        bestLoss: r.best_loss ?? undefined,
        bestAccuracy: r.best_accuracy ?? undefined,
    };
}

export type CompatibilityStatus = 'compatible' | 'adaptable' | 'incompatible';

interface GenomeLibraryState {
    entries: GenomeLibraryEntry[];
    isLoading: boolean;

    loadLibrary: () => Promise<void>;
    saveGenome: (genomeStr: string, name: string, tags: string[]) => Promise<GenomeLibraryEntry>;
    deleteGenome: (id: string) => Promise<void>;
    loadGenomeContent: (id: string) => Promise<string>;
}

export const useGenomeLibraryStore = create<GenomeLibraryState>()((set) => ({
    entries: [],
    isLoading: false,

    loadLibrary: async () => {
        set({ isLoading: true });
        try {
            const result = await invoke<RustGenomeLibraryEntry[]>('list_library_genomes');
            set({ entries: result.map(mapEntry) });
        } catch (err) {
            console.error('Failed to load genome library:', err);
        } finally {
            set({ isLoading: false });
        }
    },

    saveGenome: async (genomeStr: string, name: string, tags: string[]) => {
        const result = await invoke<RustGenomeLibraryEntry>('save_to_library', {
            genomeStr,
            name,
            tags,
        });
        const entry = mapEntry(result);
        set((state) => ({ entries: [...state.entries, entry] }));
        return entry;
    },

    deleteGenome: async (id: string) => {
        await invoke('delete_from_library', { id });
        set((state) => ({ entries: state.entries.filter((e) => e.id !== id) }));
    },

    loadGenomeContent: async (id: string) => {
        return await invoke<string>('load_library_genome', { id });
    },
}));
