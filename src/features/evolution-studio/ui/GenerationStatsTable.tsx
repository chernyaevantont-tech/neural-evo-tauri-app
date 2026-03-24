import React from 'react';
import type { GenerationSnapshot } from '../../../entities/genome';
import styles from './GenerationStatsTable.module.css';

interface GenerationStatsTableProps {
    generations: GenerationSnapshot[];
    selectedGeneration?: number;
    onSelectGeneration?: (generation: number) => void;
}

export function GenerationStatsTable({
    generations,
    selectedGeneration,
    onSelectGeneration,
}: GenerationStatsTableProps) {
    return (
        <div className={styles.tableWrap}>
            <table className={styles.table}>
                <thead>
                    <tr>
                        <th>Gen</th>
                        <th>Genomes</th>
                        <th>Best Fitness</th>
                        <th>Avg Nodes</th>
                        <th>Training Time (ms)</th>
                        <th>Avg Inference (ms)</th>
                        <th>Peak VRAM (MB)</th>
                        <th>Throughput (samples/s)</th>
                    </tr>
                </thead>
                <tbody>
                    {generations.map((gen) => {
                        const avgInference = (gen.totalInferenceMs ?? 0) / Math.max(1, gen.genomes.length);
                        const peakVram = gen.genomes.reduce(
                            (max, g) => Math.max(max, g.profiler?.peak_active_memory_mb ?? 0),
                            0,
                        );
                        const isActive = selectedGeneration === gen.generation;

                        return (
                            <tr
                                key={gen.generation}
                                className={`${styles.row} ${isActive ? styles.rowActive : ''}`}
                                onClick={() => onSelectGeneration?.(gen.generation)}
                            >
                                <td>{gen.generation}</td>
                                <td>{gen.genomes.length}</td>
                                <td>{gen.bestFitness.toFixed(3)}</td>
                                <td>{gen.avgNodes.toFixed(1)}</td>
                                <td>{(gen.totalTrainingMs ?? 0).toLocaleString()}</td>
                                <td>{avgInference.toFixed(3)}</td>
                                <td>{peakVram.toFixed(1)}</td>
                                <td>{(gen.avgSamplesPerSec ?? 0).toFixed(1)}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
