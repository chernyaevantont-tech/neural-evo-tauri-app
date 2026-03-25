import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import type { GenealogyPathDto, TrainingProfiler } from '../../../shared/lib';

export interface GenomeFitnessMetrics {
    loss: number;
    accuracy: number;
    adjustedFitness?: number;
    inferenceLatencyMs?: number;
    modelSizeMb?: number;
    trainingTimeMs?: number;
}

export interface HiddenLibraryQuery {
    generationMin?: number;
    generationMax?: number;
    accuracyMin?: number;
    accuracyMax?: number;
    latencyMinMs?: number;
    latencyMaxMs?: number;
    modelSizeMinMb?: number;
    modelSizeMaxMb?: number;
    parentGenomeId?: string;
    createdAfterUnixMs?: number;
    createdBeforeUnixMs?: number;
    limit?: number;
}

export interface WeightExportResponse {
    weightsPath: string;
    metadataPath: string;
    usedCachedWeights: boolean;
}

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
    isHidden?: boolean;
    sourceGeneration?: number;
    parentGenomes?: string[];
    fitnessMetrics?: GenomeFitnessMetrics;
    profilerData?: TrainingProfiler;
    createdAtUnixMs?: number;
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
    is_hidden?: boolean;
    source_generation?: number;
    parent_genomes?: string[];
    fitness_metrics?: {
        loss: number;
        accuracy: number;
        adjusted_fitness?: number | null;
        inference_latency_ms?: number | null;
        model_size_mb?: number | null;
        training_time_ms?: number | null;
    } | null;
    profiler_data?: TrainingProfiler | null;
    created_at_unix_ms?: number;
}

interface RustWeightExportResponse {
    weights_path: string;
    metadata_path: string;
    used_cached_weights: boolean;
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
        isHidden: r.is_hidden ?? undefined,
        sourceGeneration: r.source_generation ?? undefined,
        parentGenomes: r.parent_genomes ?? undefined,
        fitnessMetrics: r.fitness_metrics
            ? {
                loss: r.fitness_metrics.loss,
                accuracy: r.fitness_metrics.accuracy,
                adjustedFitness: r.fitness_metrics.adjusted_fitness ?? undefined,
                inferenceLatencyMs: r.fitness_metrics.inference_latency_ms ?? undefined,
                modelSizeMb: r.fitness_metrics.model_size_mb ?? undefined,
                trainingTimeMs: r.fitness_metrics.training_time_ms ?? undefined,
            }
            : undefined,
        profilerData: r.profiler_data ?? undefined,
        createdAtUnixMs: r.created_at_unix_ms ?? undefined,
    };
}

function mapWeightExportResponse(r: RustWeightExportResponse): WeightExportResponse {
    return {
        weightsPath: r.weights_path,
        metadataPath: r.metadata_path,
        usedCachedWeights: r.used_cached_weights,
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
    listHiddenLibrary: (query?: HiddenLibraryQuery) => Promise<GenomeLibraryEntry[]>;
    unhideHiddenGenome: (id: string) => Promise<void>;
    deleteHiddenGenome: (id: string) => Promise<void>;
    getGenealogyPath: (genomeId: string) => Promise<GenealogyPathDto>;
    pickFolder: () => Promise<string>;
    exportGenomeWithWeights: (genomeId: string, outputPath: string) => Promise<WeightExportResponse>;
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

    listHiddenLibrary: async (query?: HiddenLibraryQuery) => {
        const result = await invoke<RustGenomeLibraryEntry[]>('list_hidden_library', {
            query: query ?? null,
        });
        return result.map(mapEntry);
    },

    unhideHiddenGenome: async (id: string) => {
        await invoke('unhide_genome', { genomeId: id });
        set((state) => ({
            entries: state.entries.filter((e) => e.id !== id),
        }));
    },

    deleteHiddenGenome: async (id: string) => {
        await invoke('delete_hidden_genome', { genomeId: id });
        set((state) => ({
            entries: state.entries.filter((e) => e.id !== id),
        }));
    },

    getGenealogyPath: async (genomeId: string) => {
        return await invoke<GenealogyPathDto>('get_genealogy', { genomeId });
    },

    pickFolder: async () => {
        return await invoke<string>('pick_folder');
    },

    exportGenomeWithWeights: async (genomeId: string, outputPath: string) => {
        const result = await invoke<RustWeightExportResponse>('export_genome_with_weights', {
            genomeId,
            outputPath,
        });
        return mapWeightExportResponse(result);
    },
}));
