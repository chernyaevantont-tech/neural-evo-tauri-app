import type { BaseNode, Genome } from '../../canvas-genome';
import type {
    GenomeGenealogy,
    GenomeObjectives,
    MutationType,
    TrainingProfiler,
} from '../../../shared/lib';

export interface ZeroCostMetricsView {
    synflow: number;
    normalized_score: number;
    strategy_decision: 'skip' | 'partial_train' | 'full_train';
}

export interface BatchMetrics {
    epoch: number;
    batch: number;
    total_batches: number;
    loss: number;
    accuracy: number;
}

export interface PopulatedGenome {
    id: string;
    genome: Genome;
    nodes: BaseNode[];
    loss?: number;
    accuracy?: number;
    adjustedFitness?: number;
    trainingMetrics?: BatchMetrics[];
    resources?: { totalFlash: number; totalRam: number; totalMacs: number; totalNodes: number };
    zeroCostMetric?: ZeroCostMetricsView;
    profiler?: TrainingProfiler;
    objectives?: GenomeObjectives;
    is_dominated?: boolean;
    generation?: number;
    parent_ids?: string[];
    mutation_type?: MutationType;
    mutation_params?: Record<string, unknown>;
}

export interface GenerationSnapshot {
    generation: number;
    genomes: PopulatedGenome[];
    bestFitness: number;
    avgNodes: number;
    timestamp: string;
    evaluated: boolean;
    genealogy?: Map<string, GenomeGenealogy>;
    paretoFront?: GenomeObjectives[];
    objectiveSpace?: {
        accuracy: { min: number; max: number };
        latency: { min: number; max: number };
        modelSize: { min: number; max: number };
    };
    totalTrainingMs?: number;
    totalInferenceMs?: number;
    avgSamplesPerSec?: number;
}
