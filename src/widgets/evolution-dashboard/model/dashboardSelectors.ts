import type { BatchMetrics, GenerationSnapshot, PopulatedGenome } from '../../../entities/genome';
import type { GenerationParetoFront } from '../../../shared/lib';

export type DashboardJobStatus = 'running' | 'completed' | 'failed' | 'queued';
export type DashboardJobStage = 'train' | 'val' | 'test';

export interface DashboardJob {
    jobId: string;
    genomeId: string;
    stage: DashboardJobStage;
    progressPercent: number;
    status: DashboardJobStatus;
    durationMs: number;
    etaSeconds?: number;
}

export interface OverviewMetrics {
    generationsElapsed: number;
    genomesEvaluated: number;
    currentBestFitness?: number;
    paretoFrontSize: number;
    elapsedTimeSeconds: number;
    etaSeconds?: number;
    feasibleSolutions: number;
    feasibleTotal: number;
}

export interface FitnessPoint {
    generation: number;
    bestFitness: number;
    avgFitness: number;
    feasibleFrontSize?: number;
    constraintPressure?: number;
}

const UNKNOWN_VALUE = 'data unavailable';

export function formatDuration(totalSeconds: number): string {
    if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
        return UNKNOWN_VALUE;
    }

    const seconds = Math.floor(totalSeconds % 60);
    const minutes = Math.floor((totalSeconds / 60) % 60);
    const hours = Math.floor(totalSeconds / 3600);

    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

export function formatEta(totalSeconds?: number): string {
    if (totalSeconds === undefined || !Number.isFinite(totalSeconds) || totalSeconds < 0) {
        return UNKNOWN_VALUE;
    }

    return formatDuration(totalSeconds);
}

export function formatMemory(memoryMb?: number): string {
    if (memoryMb === undefined || !Number.isFinite(memoryMb) || memoryMb < 0) {
        return UNKNOWN_VALUE;
    }

    if (memoryMb >= 1024) {
        return `${(memoryMb / 1024).toFixed(2)} GB`;
    }

    return `${memoryMb.toFixed(1)} MB`;
}

export function deriveOverviewMetrics(params: {
    generation: number;
    generationHistory: GenerationSnapshot[];
    elapsedTimeSeconds: number;
    useMaxGenerations: boolean;
    maxGenerations: number;
    currentParetoFrontSize: number;
    feasibleByGenomeId: Record<string, boolean>;
}): OverviewMetrics {
    const {
        generation,
        generationHistory,
        elapsedTimeSeconds,
        useMaxGenerations,
        maxGenerations,
        currentParetoFrontSize,
        feasibleByGenomeId,
    } = params;

    const evaluatedSnapshots = generationHistory.filter((snapshot) => snapshot.evaluated);
    const genomesEvaluated = evaluatedSnapshots.reduce((sum, snapshot) => sum + snapshot.genomes.length, 0);

    const currentBestFitness = evaluatedSnapshots.length > 0
        ? evaluatedSnapshots[evaluatedSnapshots.length - 1].bestFitness
        : undefined;

    const feasibleGenomeIds = Object.keys(feasibleByGenomeId).filter((id) => feasibleByGenomeId[id]);
    const feasibleSolutions = feasibleGenomeIds.length;
    const feasibleTotal = Object.keys(feasibleByGenomeId).length;

    let etaSeconds: number | undefined;
    if (useMaxGenerations && generation > 0 && elapsedTimeSeconds > 0 && maxGenerations > generation) {
        const avgGenSeconds = elapsedTimeSeconds / generation;
        etaSeconds = Math.max(0, Math.round(avgGenSeconds * (maxGenerations - generation)));
    }

    return {
        generationsElapsed: generation,
        genomesEvaluated,
        currentBestFitness,
        paretoFrontSize: currentParetoFrontSize,
        elapsedTimeSeconds,
        etaSeconds,
        feasibleSolutions,
        feasibleTotal,
    };
}

export function buildFitnessTimeline(
    generationHistory: GenerationSnapshot[],
    paretoHistory: Map<number, GenerationParetoFront>,
    feasibilityByGenomeId: Record<string, boolean>,
    constraintViolationScoreByGenomeId: Record<string, number>,
): FitnessPoint[] {
    return generationHistory
        .filter((snapshot) => snapshot.evaluated)
        .map((snapshot) => {
            const avgFitness = snapshot.genomes.length > 0
                ? snapshot.genomes.reduce((sum, genome) => sum + (genome.adjustedFitness ?? 0), 0) / snapshot.genomes.length
                : 0;

            const pareto = paretoHistory.get(snapshot.generation);
            const members = pareto?.pareto_members ?? [];
            const feasibleFrontSize = members.filter((m) => feasibilityByGenomeId[m.genome_id]).length;

            const violationScores = members
                .map((m) => constraintViolationScoreByGenomeId[m.genome_id])
                .filter((score): score is number => Number.isFinite(score));

            const constraintPressure = violationScores.length > 0
                ? violationScores.reduce((sum, score) => sum + score, 0) / violationScores.length
                : undefined;

            return {
                generation: snapshot.generation,
                bestFitness: snapshot.bestFitness,
                avgFitness,
                feasibleFrontSize,
                constraintPressure,
            };
        });
}

export function buildJobs(params: {
    isRunning: boolean;
    currentEvaluatingIndex: number;
    liveMetrics: BatchMetrics[];
    population: PopulatedGenome[];
    currentSnapshot?: GenerationSnapshot;
}): DashboardJob[] {
    const { isRunning, currentEvaluatingIndex, liveMetrics, population, currentSnapshot } = params;
    const source = currentSnapshot?.genomes.length ? currentSnapshot.genomes : population;

    const latestMetric = liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1] : undefined;
    const hasIndexedMetrics = liveMetrics.some((metric) => typeof metric.genome_index === 'number');
    const latestByGenome = new Map<number, BatchMetrics>();
    for (const metric of liveMetrics) {
        const idx = metric.genome_index;
        if (typeof idx !== 'number') {
            continue;
        }
        const prev = latestByGenome.get(idx);
        const metricStep = metric.step ?? metric.batch;
        const prevStep = prev?.step ?? prev?.batch ?? 0;
        if (!prev || metricStep >= prevStep) {
            latestByGenome.set(idx, metric);
        }
    }

    return source.map((genome, index) => {
        let status: DashboardJobStatus = 'queued';
        let progressPercent = 0;
        let etaSeconds: number | undefined;
        const metricForGenome = latestByGenome.get(index);
        const evaluated = genome.profiler !== undefined || genome.accuracy !== undefined || genome.loss !== undefined;

        if (!isRunning) {
            status = evaluated ? 'completed' : 'failed';
            progressPercent = evaluated ? 100 : 0;
        } else if (evaluated) {
            status = 'completed';
            progressPercent = 100;
        } else if (hasIndexedMetrics) {
            if (metricForGenome) {
                status = 'running';
                const doneSteps = metricForGenome.step ?? metricForGenome.batch;
                const totalSteps = metricForGenome.total_steps
                    ?? (metricForGenome.total_batches * Math.max(1, metricForGenome.epoch));
                progressPercent = Math.round((doneSteps / Math.max(1, totalSteps)) * 100);

                const elapsedMs = metricForGenome.elapsed_train_ms ?? 0;
                const queueWaitMs = metricForGenome.queue_wait_ms ?? 0;
                const observedStepMs = metricForGenome.step_time_ms ?? 0;
                if (doneSteps > 0 && (elapsedMs > 0 || observedStepMs > 0)) {
                    const avgPerStepMs = observedStepMs > 0 ? observedStepMs : elapsedMs / doneSteps;
                    const remainingSteps = Math.max(0, totalSteps - doneSteps);
                    etaSeconds = Math.max(0, Math.round(((avgPerStepMs * remainingSteps) + queueWaitMs) / 1000));
                }
            }
        } else if (index < currentEvaluatingIndex) {
            status = 'completed';
            progressPercent = 100;
        } else if (index === currentEvaluatingIndex) {
            status = 'running';
            progressPercent = latestMetric
                ? Math.round((latestMetric.batch / Math.max(1, latestMetric.total_batches)) * 100)
                : 0;

            if (latestMetric && latestMetric.batch > 0) {
                const elapsedMs = latestMetric.elapsed_train_ms ?? 0;
                const queueWaitMs = latestMetric.queue_wait_ms ?? 0;
                const observedStepMs = latestMetric.step_time_ms ?? 0;
                if (elapsedMs > 0 || observedStepMs > 0) {
                    const avgPerBatchMs = observedStepMs > 0 ? observedStepMs : elapsedMs / latestMetric.batch;
                    const remainingBatches = Math.max(0, latestMetric.total_batches - latestMetric.batch);
                    etaSeconds = Math.round(((avgPerBatchMs * remainingBatches) + queueWaitMs) / 1000);
                }
            }
        }

        return {
            jobId: `job-${index + 1}`,
            genomeId: genome.id,
            stage: status === 'completed' ? 'test' : status === 'running' ? 'train' : 'val',
            progressPercent,
            status,
            durationMs: genome.profiler?.total_train_duration_ms ?? 0,
            etaSeconds,
        };
    });
}
