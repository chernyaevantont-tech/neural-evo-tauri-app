import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    Tooltip,
    Legend,
    type ChartData,
    type ChartOptions,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import type { GenerationParetoFront, GenomeObjectives } from '../../shared/lib';
import styles from './pareto-front-visualizer.module.css';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

type ParetoPoint = {
    x: number;
    y: number;
    genomeId: string;
    accuracy: number;
    latencyMs: number;
    modelSizeMb: number;
    trainTimeMs: number;
    feasible?: boolean;
    violationScore?: number;
};

type Props = {
    pareto: GenerationParetoFront;
    selectedGenomeId?: string;
    onSelectGenome: (genomeId: string) => void;
    feasibilityByGenomeId?: Record<string, boolean>;
    constraintViolationScoreByGenomeId?: Record<string, number>;
};

function clamp01(value: number): number {
    return Math.max(0, Math.min(1, value));
}

function colorByFeasibility(point: ParetoPoint, frontier: boolean): string {
    if (point.feasible === true) {
        return frontier ? 'rgba(34, 197, 94, 0.95)' : 'rgba(34, 197, 94, 0.58)';
    }

    if (point.feasible === false) {
        const score = clamp01((point.violationScore ?? 0) / 1.5);
        const alphaBase = frontier ? 0.35 : 0.25;
        const alpha = alphaBase + score * 0.55;
        return `rgba(239, 68, 68, ${alpha.toFixed(3)})`;
    }

    return frontier ? 'rgba(59, 130, 246, 0.8)' : 'rgba(148, 163, 184, 0.4)';
}

function toPoint(
    objective: GenomeObjectives,
    feasibilityByGenomeId?: Record<string, boolean>,
    constraintViolationScoreByGenomeId?: Record<string, number>,
): ParetoPoint {
    return {
        x: objective.inference_latency_ms,
        y: objective.accuracy,
        genomeId: objective.genome_id,
        accuracy: objective.accuracy,
        latencyMs: objective.inference_latency_ms,
        modelSizeMb: objective.model_size_mb,
        trainTimeMs: objective.training_time_ms ?? objective.train_time_ms ?? 0,
        feasible: feasibilityByGenomeId?.[objective.genome_id] ?? objective.device_feasible,
        violationScore:
            constraintViolationScoreByGenomeId?.[objective.genome_id] ??
            objective.constraint_violation_score,
    };
}

export function ParetoScatterPlot({
    pareto,
    selectedGenomeId,
    onSelectGenome,
    feasibilityByGenomeId,
    constraintViolationScoreByGenomeId,
}: Props) {
    const allObjectives = pareto.all_genomes ?? pareto.pareto_members;
    const frontierSet = useMemo(
        () =>
            new Set(
                pareto.frontier_genome_ids ?? pareto.pareto_members.map((item) => item.genome_id),
            ),
        [pareto.frontier_genome_ids, pareto.pareto_members],
    );

    const { dominatedPoints, frontierPoints } = useMemo(() => {
        const dominated: ParetoPoint[] = [];
        const frontier: ParetoPoint[] = [];

        for (const objective of allObjectives) {
            const point = toPoint(objective, feasibilityByGenomeId, constraintViolationScoreByGenomeId);
            if (frontierSet.has(objective.genome_id)) {
                frontier.push(point);
            } else {
                dominated.push(point);
            }
        }

        return { dominatedPoints: dominated, frontierPoints: frontier };
    }, [
        allObjectives,
        constraintViolationScoreByGenomeId,
        feasibilityByGenomeId,
        frontierSet,
    ]);

    const data: ChartData<'scatter', ParetoPoint[]> = {
        datasets: [
            {
                label: 'Dominated',
                data: dominatedPoints,
                backgroundColor: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return colorByFeasibility(point, false);
                },
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBorderColor: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return point?.genomeId === selectedGenomeId ? '#f59e0b' : 'rgba(148, 163, 184, 0.8)';
                },
                pointBorderWidth: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return point?.genomeId === selectedGenomeId ? 3 : 1;
                },
            },
            {
                label: 'Non-dominated frontier',
                data: frontierPoints,
                backgroundColor: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return colorByFeasibility(point, true);
                },
                pointBorderColor: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return point?.genomeId === selectedGenomeId ? '#f59e0b' : '#14532d';
                },
                pointBorderWidth: (ctx) => {
                    const point = ctx.raw as ParetoPoint;
                    return point?.genomeId === selectedGenomeId ? 3 : 2;
                },
                pointRadius: 6,
                pointHoverRadius: 8,
            },
        ],
    };

    const options: ChartOptions<'scatter'> = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'nearest',
            intersect: true,
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Inference Latency (ms)',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Accuracy',
                },
            },
        },
        plugins: {
            legend: {
                labels: {
                    usePointStyle: true,
                },
            },
            tooltip: {
                callbacks: {
                    label: (ctx) => {
                        const point = ctx.raw as ParetoPoint;
                        const feasibleLabel =
                            point.feasible === undefined
                                ? 'n/a'
                                : point.feasible
                                    ? 'yes'
                                    : 'no';
                        const violationLabel =
                            point.violationScore === undefined
                                ? 'n/a'
                                : point.violationScore.toFixed(3);

                        return [
                            `Genome: ${point.genomeId}`,
                            `Accuracy: ${point.accuracy.toFixed(4)}`,
                            `Latency: ${point.latencyMs.toFixed(3)} ms`,
                            `Model size: ${point.modelSizeMb.toFixed(3)} MB`,
                            `Train time: ${point.trainTimeMs.toFixed(1)} ms`,
                            `Device feasible: ${feasibleLabel}`,
                            `Constraint score: ${violationLabel}`,
                        ];
                    },
                },
            },
        },
        onClick: (_event, elements, chart) => {
            if (!elements.length) {
                return;
            }
            const first = elements[0];
            const selected = chart.data.datasets[first.datasetIndex]?.data?.[first.index] as
                | ParetoPoint
                | undefined;
            if (selected?.genomeId) {
                onSelectGenome(selected.genomeId);
            }
        },
    };

    if (allObjectives.length === 0) {
        return (
            <div className={styles.emptyState}>
                Pareto data is empty for the selected view.
            </div>
        );
    }

    return (
        <div className={styles.scatterWrap} data-testid="pareto-scatter-wrap">
            <Scatter data={data} options={options} />
        </div>
    );
}