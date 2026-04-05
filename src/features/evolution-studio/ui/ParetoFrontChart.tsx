import React, { useState, useMemo } from 'react';
import { Scatter } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    ScatterController,
    PointElement,
    LinearScale,
    Tooltip,
    Legend,
    ChartOptions,
    ChartData
} from 'chart.js';
import { PopulatedGenome } from '../model/useEvolutionLoop';

ChartJS.register(ScatterController, PointElement, LinearScale, Tooltip, Legend);

type ObjectiveKey = 'quality' | 'flashKB' | 'ramKB' | 'macs';

interface ParetoFrontChartProps {
    paretoFront: PopulatedGenome[];
    onGenomeSelect?: (genome: PopulatedGenome) => void;
    selectedGenomeId?: string;
}

interface ObjectiveAxis {
    key: ObjectiveKey;
    label: string;
    reverse: boolean;  // true for resources (lower is better)
}

const AXES: Record<ObjectiveKey, ObjectiveAxis> = {
    quality: { key: 'quality', label: 'Quality (Accuracy/Proxy)', reverse: false },
    flashKB: { key: 'flashKB', label: 'Flash Memory (KB)', reverse: true },
    ramKB: { key: 'ramKB', label: 'RAM (KB)', reverse: true },
    macs: { key: 'macs', label: 'MACs (Millions)', reverse: true }
};

export const ParetoFrontChart: React.FC<ParetoFrontChartProps> = ({
    paretoFront,
    onGenomeSelect,
    selectedGenomeId
}) => {
    const [xAxis, setXAxis] = useState<ObjectiveKey>('flashKB');
    const [yAxis, setYAxis] = useState<ObjectiveKey>('quality');

    // Prepare chart data
    const chartData: ChartData<'scatter'> = useMemo(() => {
        const data = paretoFront.map(g => {
            if (!g.objectives) return null;
            return {
                x: g.objectives[xAxis] || 0,
                y: g.objectives[yAxis] || 0,
                genome: g
            };
        }).filter((p): p is NonNullable<typeof p> => p !== null);

        return {
            datasets: [{
                label: 'Pareto-Optimal Solutions',
                data,
                backgroundColor: data.map(p => 
                    p.genome.id === selectedGenomeId 
                        ? 'rgba(255, 99, 132, 0.8)'  // Highlighted
                        : 'rgba(74, 222, 128, 0.6)'   // Normal
                ),
                borderColor: data.map(p => 
                    p.genome.id === selectedGenomeId 
                        ? 'rgba(255, 99, 132, 1)'
                        : 'rgba(74, 222, 128, 1)'
                ),
                pointRadius: data.map(p => 
                    p.genome.id === selectedGenomeId ? 10 : 6
                ),
                pointHoverRadius: 8
            }]
        };
    }, [paretoFront, xAxis, yAxis, selectedGenomeId]);

    const options: ChartOptions<'scatter'> = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        const g = (context.raw as any)?.genome as PopulatedGenome;
                        if (!g?.objectives) return [];
                        return [
                            `Quality: ${(g.objectives.quality * 100).toFixed(1)}%`,
                            `Flash: ${g.objectives.flashKB.toFixed(1)} KB`,
                            `RAM: ${g.objectives.ramKB.toFixed(1)} KB`,
                            `MACs: ${g.objectives.macs.toFixed(2)} M`,
                            `Nodes: ${g.nodes.length}`,
                            `Pareto Rank: ${g.paretoRank || 0}`
                        ];
                    }
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: AXES[xAxis].label,
                    font: { size: 14, weight: 'bold' }
                },
                reverse: AXES[xAxis].reverse,
                beginAtZero: true
            },
            y: {
                title: {
                    display: true,
                    text: AXES[yAxis].label,
                    font: { size: 14, weight: 'bold' }
                },
                reverse: AXES[yAxis].reverse,
                beginAtZero: true
            }
        },
        onClick: (_, elements) => {
            if (elements.length > 0 && onGenomeSelect) {
                const dataIndex = elements[0].index;
                const genome = chartData.datasets[0].data[dataIndex] as any;
                if (genome?.genome) {
                    onGenomeSelect(genome.genome);
                }
            }
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
            {/* Axis Controls */}
            <div style={{ 
                display: 'flex', 
                gap: '1rem', 
                alignItems: 'center',
                padding: '0.5rem',
                background: 'var(--color-bg-secondary)',
                borderRadius: '8px'
            }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontWeight: 500 }}>X-Axis:</span>
                    <select
                        value={xAxis}
                        onChange={e => setXAxis(e.target.value as ObjectiveKey)}
                        style={{
                            padding: '0.5rem',
                            borderRadius: '4px',
                            border: '1px solid var(--color-border)',
                            background: 'var(--color-bg-primary)',
                            color: 'var(--color-text)'
                        }}
                    >
                        {Object.entries(AXES).map(([key, axis]) => (
                            <option key={key} value={key}>{axis.label}</option>
                        ))}
                    </select>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontWeight: 500 }}>Y-Axis:</span>
                    <select
                        value={yAxis}
                        onChange={e => setYAxis(e.target.value as ObjectiveKey)}
                        style={{
                            padding: '0.5rem',
                            borderRadius: '4px',
                            border: '1px solid var(--color-border)',
                            background: 'var(--color-bg-primary)',
                            color: 'var(--color-text)'
                        }}
                    >
                        {Object.entries(AXES).map(([key, axis]) => (
                            <option key={key} value={key}>{axis.label}</option>
                        ))}
                    </select>
                </label>

                <div style={{ marginLeft: 'auto', color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
                    {paretoFront.length} solutions on Pareto front
                </div>
            </div>

            {/* Chart */}
            <div style={{ flex: 1, minHeight: '400px' }}>
                {paretoFront.length > 0 ? (
                    <Scatter data={chartData} options={options} />
                ) : (
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        color: 'var(--color-text-muted)'
                    }}>
                        No Pareto-optimal solutions yet. Start evolution to discover trade-offs.
                    </div>
                )}
            </div>

            {/* Info */}
            <div style={{
                padding: '0.75rem',
                background: 'var(--color-bg-secondary)',
                borderRadius: '8px',
                fontSize: '0.875rem',
                color: 'var(--color-text-muted)'
            }}>
                <strong>Tip:</strong> Each point represents a Pareto-optimal architecture. 
                Points on the bottom-right (for X=resources, Y=quality) offer the best trade-off. 
                Click a point to inspect the architecture.
            </div>
        </div>
    );
};
