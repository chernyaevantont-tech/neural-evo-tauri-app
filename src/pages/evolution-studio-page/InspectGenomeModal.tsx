import React from 'react';
import styles from './InspectGenomeModal.module.css';
import { BaseNode, GenomeSvgPreview } from '../../entities/canvas-genome';
import { BsX } from 'react-icons/bs';
import { BatchMetrics } from '../../features/evolution-studio/model/useEvolutionLoop';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ChartOptions,
    ChartData
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface InspectGenomeModalProps {
    title: string;
    subtitle?: string;
    nodes: BaseNode[];
    trainingMetrics?: BatchMetrics[];
    onClose: () => void;
}

export const InspectGenomeModal: React.FC<InspectGenomeModalProps> = ({ title, subtitle, nodes, trainingMetrics, onClose }) => {
    const hasMetrics = trainingMetrics && trainingMetrics.length > 0;

    const chartData: ChartData<'line'> | null = hasMetrics ? {
        labels: trainingMetrics.map(m => m.batch.toString()),
        datasets: [
            {
                label: 'Loss',
                yAxisID: 'y',
                data: trainingMetrics.map(m => m.loss),
                borderColor: '#ffb86c',
                backgroundColor: 'rgba(255, 184, 108, 0.15)',
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Accuracy (%)',
                yAxisID: 'y1',
                data: trainingMetrics.map(m => m.accuracy),
                borderColor: '#50fa7b',
                backgroundColor: 'rgba(80, 250, 123, 0.15)',
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2,
            }
        ]
    } : null;

    const chartOptions: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
            x: { display: false },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: { display: true, text: 'Loss', color: '#999' },
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255, 255, 255, 0.08)' }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: { display: true, text: 'Accuracy %', color: '#999' },
                ticks: { color: '#aaa' },
                min: 0,
                max: 100,
                grid: { drawOnChartArea: false },
            }
        },
        plugins: {
            legend: { position: 'top', labels: { color: '#ccc' } },
            tooltip: {
                callbacks: {
                    title: (context) => {
                        const idx = context[0].dataIndex;
                        const m = trainingMetrics![idx];
                        return `Epoch: ${m.epoch} | Batch: ${m.batch} / ${m.total_batches}`;
                    }
                }
            }
        }
    };

    return (
        <div className={styles.overlay} onClick={onClose}>
            <div className={styles.modal} onClick={e => e.stopPropagation()}>
                <div className={styles.header}>
                    <div className={styles.titles}>
                        <h2 className={styles.title}>{title}</h2>
                        {subtitle && <div className={styles.subtitle}>{subtitle}</div>}
                    </div>
                    <button className={styles.closeBtn} onClick={onClose}><BsX /></button>
                </div>
                <div className={styles.content}>
                    <GenomeSvgPreview nodes={nodes} />
                    {chartData && (
                        <div style={{ width: '100%', height: '200px', marginTop: '12px' }}>
                            <Line data={chartData} options={chartOptions} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
