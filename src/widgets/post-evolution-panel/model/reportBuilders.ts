import type { PopulatedGenome } from '../../../entities/genome';
import type { DeviceConstraintParams } from '../../../features/evolution-manager';
import type {
    GenerationParetoFront,
    GenomeGenealogy,
    GenomeObjectives,
    StoppingCriterionType,
} from '../../../shared/lib';

export interface ComparisonRow {
    genomeId: string;
    architectureSummary: string;
    accuracy: number;
    latencyMs: number;
    modelSizeMb: number;
    trainingTimeMs: number;
    inferenceTimeMs: number;
    memoryBreakdown: string;
    lineageDepth: number;
    deviceRatios: {
        mops: number;
        ram: number;
        flash: number;
        latency: number;
    };
}

export interface HiddenArchiveSummary {
    total: number;
    avgFitness: number;
    minAccuracy: number;
    maxAccuracy: number;
    feasibleCount: number;
}

export interface EvolutionReportDataModel {
    generatedAtIso: string;
    runConfig: {
        generation: number;
        elapsedRuntimeSeconds: number;
        stoppingPolicy: StoppingCriterionType[];
    };
    topGenomes: Array<{
        genome_id: string;
        accuracy: number;
        latency_ms: number;
        model_size_mb: number;
        feasible?: boolean;
    }>;
    paretoSummary: {
        generation: number;
        pareto_count: number;
        total_genomes: number;
    };
    constraintsSummary?: {
        mops_budget: number;
        ram_mb: number;
        flash_mb: number;
        latency_budget_ms: number;
    };
    stoppingReason: string;
    hiddenArchive: HiddenArchiveSummary;
}

function ratio(numerator: number, denominator?: number): number {
    if (!denominator || denominator <= 0) {
        return Number.NaN;
    }
    return numerator / denominator;
}

function finiteOrZero(value?: number): number {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return 0;
    }
    return value;
}

function summarizeArchitecture(genome?: PopulatedGenome): string {
    if (!genome || genome.nodes.length === 0) {
        return 'N/A';
    }

    const byType = new Map<string, number>();
    for (const node of genome.nodes) {
        const type = typeof node.GetNodeType === 'function' ? node.GetNodeType() : 'Unknown';
        byType.set(type, (byType.get(type) ?? 0) + 1);
    }

    const top = Array.from(byType.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([type, count]) => `${type} x${count}`)
        .join(', ');

    return `${genome.nodes.length} nodes | ${top}`;
}

function buildLineageDepth(genomeId: string, tree?: Map<string, GenomeGenealogy>): number {
    if (!tree || tree.size === 0) {
        return 0;
    }

    const memo = new Map<string, number>();

    const walk = (id: string): number => {
        if (memo.has(id)) {
            return memo.get(id) ?? 0;
        }

        const node = tree.get(id);
        if (!node || node.parent_ids.length === 0) {
            memo.set(id, 0);
            return 0;
        }

        const depth = 1 + Math.max(...node.parent_ids.map((parentId) => walk(parentId)));
        memo.set(id, depth);
        return depth;
    };

    return walk(genomeId);
}

export function buildComparisonRows(params: {
    selectedGenomeIds: string[];
    objectivesByGenomeId: Map<string, GenomeObjectives>;
    genomeById: Map<string, PopulatedGenome>;
    genealogyTree?: Map<string, GenomeGenealogy>;
    activeDeviceConstraints?: DeviceConstraintParams;
}): ComparisonRow[] {
    const {
        selectedGenomeIds,
        objectivesByGenomeId,
        genomeById,
        genealogyTree,
        activeDeviceConstraints,
    } = params;

    return selectedGenomeIds.map((genomeId) => {
        const objective = objectivesByGenomeId.get(genomeId);
        const genome = genomeById.get(genomeId);

        const ramUsedMb = finiteOrZero(genome?.resources?.totalRam) / (1024 * 1024);
        const mopsUsed = finiteOrZero(genome?.resources?.totalMacs) / 1_000_000;
        const modelSizeMb =
            objective?.model_size_mb ??
            (finiteOrZero(genome?.resources?.totalFlash) > 0
                ? finiteOrZero(genome?.resources?.totalFlash) / (1024 * 1024)
                : 0);
        const latencyMs =
            objective?.inference_latency_ms ??
            finiteOrZero(genome?.profiler?.inference_msec_per_sample);

        return {
            genomeId,
            architectureSummary: summarizeArchitecture(genome),
            accuracy: finiteOrZero(objective?.accuracy ?? genome?.accuracy),
            latencyMs: finiteOrZero(latencyMs),
            modelSizeMb: finiteOrZero(modelSizeMb),
            trainingTimeMs: finiteOrZero(objective?.training_time_ms ?? genome?.profiler?.total_train_duration_ms),
            inferenceTimeMs: finiteOrZero(genome?.profiler?.inference_msec_per_sample),
            memoryBreakdown: genome?.profiler
                ? `model ${genome.profiler.peak_model_params_mb.toFixed(1)} MB | grad ${genome.profiler.peak_gradient_mb.toFixed(1)} MB | act ${genome.profiler.peak_activation_mb.toFixed(1)} MB`
                : 'N/A',
            lineageDepth: buildLineageDepth(genomeId, genealogyTree),
            deviceRatios: {
                mops: ratio(mopsUsed, activeDeviceConstraints?.mops_budget),
                ram: ratio(ramUsedMb, activeDeviceConstraints?.ram_mb),
                flash: ratio(finiteOrZero(modelSizeMb), activeDeviceConstraints?.flash_mb),
                latency: ratio(finiteOrZero(latencyMs), activeDeviceConstraints?.latency_budget_ms),
            },
        };
    });
}

export function buildHiddenArchiveSummary(
    objectives: GenomeObjectives[],
    feasibleByGenomeId: Record<string, boolean>,
): HiddenArchiveSummary {
    if (objectives.length === 0) {
        return {
            total: 0,
            avgFitness: 0,
            minAccuracy: 0,
            maxAccuracy: 0,
            feasibleCount: 0,
        };
    }

    const accuracies = objectives.map((item) => item.accuracy);
    const avgFitness = objectives.reduce((acc, item) => acc + item.accuracy, 0) / objectives.length;

    return {
        total: objectives.length,
        avgFitness,
        minAccuracy: Math.min(...accuracies),
        maxAccuracy: Math.max(...accuracies),
        feasibleCount: objectives.filter((item) => feasibleByGenomeId[item.genome_id]).length,
    };
}

export function buildParetoExportPayload(front: GenerationParetoFront): string {
    return JSON.stringify(
        {
            generation: front.generation,
            exported_at_iso: new Date().toISOString(),
            pareto_members: front.pareto_members,
            all_genomes: front.all_genomes ?? front.pareto_members,
            frontier_genome_ids:
                front.frontier_genome_ids ?? front.pareto_members.map((item) => item.genome_id),
        },
        null,
        2,
    );
}

export function buildEvolutionReportDataModel(params: {
    generation: number;
    elapsedRuntimeSeconds: number;
    stoppingPolicy: StoppingCriterionType[];
    stoppingReason: string;
    paretoFront: GenerationParetoFront;
    constraints?: DeviceConstraintParams;
    hiddenArchive: HiddenArchiveSummary;
}): EvolutionReportDataModel {
    const {
        generation,
        elapsedRuntimeSeconds,
        stoppingPolicy,
        stoppingReason,
        paretoFront,
        constraints,
        hiddenArchive,
    } = params;

    return {
        generatedAtIso: new Date().toISOString(),
        runConfig: {
            generation,
            elapsedRuntimeSeconds,
            stoppingPolicy,
        },
        topGenomes: paretoFront.pareto_members.slice(0, 10).map((item) => ({
            genome_id: item.genome_id,
            accuracy: item.accuracy,
            latency_ms: item.inference_latency_ms,
            model_size_mb: item.model_size_mb,
            feasible: item.device_feasible,
        })),
        paretoSummary: {
            generation: paretoFront.generation,
            pareto_count: paretoFront.pareto_members.length,
            total_genomes: (paretoFront.all_genomes ?? paretoFront.pareto_members).length,
        },
        constraintsSummary: constraints
            ? {
                mops_budget: constraints.mops_budget,
                ram_mb: constraints.ram_mb,
                flash_mb: constraints.flash_mb,
                latency_budget_ms: constraints.latency_budget_ms,
            }
            : undefined,
        stoppingReason,
        hiddenArchive,
    };
}

export function buildLineageExport(
    genealogyTree: Map<string, GenomeGenealogy> | undefined,
    format: 'json' | 'graphml',
): string {
    if (!genealogyTree || genealogyTree.size === 0) {
        return format === 'json' ? '[]' : '<graphml><graph edgedefault="directed"></graph></graphml>';
    }

    const nodes = Array.from(genealogyTree.values());

    if (format === 'json') {
        return JSON.stringify(nodes, null, 2);
    }

    const nodeXml = nodes
        .map((node) => `<node id="${node.genome_id}"/>`)
        .join('');

    const edgeXml = nodes
        .flatMap((node) => node.parent_ids.map((parentId) => `<edge source="${parentId}" target="${node.genome_id}"/>`))
        .join('');

    return `<?xml version="1.0" encoding="UTF-8"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="directed">${nodeXml}${edgeXml}</graph></graphml>`;
}
