import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    Tooltip,
    Legend,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import type { PopulatedGenome } from '../../entities/genome';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

interface ComparisonChartsProps {
    genomes: PopulatedGenome[];
}

export function ComparisonCharts({ genomes }: ComparisonChartsProps) {
    const accuracyVsTime = useMemo(
        () =>
            genomes
                .filter((g) => g.profiler)
                .map((g) => ({
                    x: g.profiler?.total_train_duration_ms ?? 0,
                    y: g.accuracy ?? 0,
                    label: g.id,
                })),
        [genomes],
    );

    const accuracyVsMemory = useMemo(
        () =>
            genomes
                .filter((g) => g.profiler)
                .map((g) => ({
                    x: g.profiler?.peak_active_memory_mb ?? 0,
                    y: g.accuracy ?? 0,
                    label: g.id,
                })),
        [genomes],
    );

    const commonTooltip = {
        callbacks: {
            label: (ctx: any) => {
                const point = ctx.raw as { x: number; y: number; label: string };
                return `${point.label.slice(0, 8)} | x=${point.x.toFixed(2)} y=${point.y.toFixed(2)}`;
            },
        },
    };

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            <div style={{ height: 240, border: '1px solid var(--color-border-primary)', borderRadius: 8, padding: '0.6rem' }}>
                <h4 style={{ margin: '0 0 0.6rem', color: 'var(--color-text-secondary)' }}>Accuracy vs Training Time</h4>
                <Scatter
                    data={{ datasets: [{ label: 'Genomes', data: accuracyVsTime, backgroundColor: 'rgba(56, 189, 248, 0.65)' }] }}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { tooltip: commonTooltip },
                        scales: {
                            x: { title: { display: true, text: 'Training Time (ms)' } },
                            y: { title: { display: true, text: 'Accuracy (%)' } },
                        },
                    }}
                />
            </div>

            <div style={{ height: 240, border: '1px solid var(--color-border-primary)', borderRadius: 8, padding: '0.6rem' }}>
                <h4 style={{ margin: '0 0 0.6rem', color: 'var(--color-text-secondary)' }}>Accuracy vs Peak Memory</h4>
                <Scatter
                    data={{ datasets: [{ label: 'Genomes', data: accuracyVsMemory, backgroundColor: 'rgba(250, 204, 21, 0.7)' }] }}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { tooltip: commonTooltip },
                        scales: {
                            x: { title: { display: true, text: 'Peak Memory (MB)' } },
                            y: { title: { display: true, text: 'Accuracy (%)' } },
                        },
                    }}
                />
            </div>
        </div>
    );
}
