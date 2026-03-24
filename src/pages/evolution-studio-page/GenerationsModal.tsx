import React, { useMemo, useState } from 'react';
import { BsX } from 'react-icons/bs';
import type { GenerationSnapshot } from '../../entities/genome';
import styles from './GenerationsModal.module.css';

type SortKey = 'evalOrder' | 'accuracy' | 'fitness' | 'nodes';

interface GenerationsModalProps {
    generations: GenerationSnapshot[];
    selectedGeneration?: number;
    onSelectGeneration: (gen: number) => void;
    onClose: () => void;
}

export const GenerationsModal: React.FC<GenerationsModalProps> = ({
    generations,
    selectedGeneration,
    onSelectGeneration,
    onClose,
}) => {
    const [sortKey, setSortKey] = useState<SortKey>('evalOrder');
    const [sortAsc, setSortAsc] = useState(false);

    const sortedGenerations = useMemo(() => {
        const gens = [...generations];
        gens.sort((a, b) => {
            let aVal: number;
            let bVal: number;

            if (sortKey === 'evalOrder') {
                aVal = a.generation;
                bVal = b.generation;
            } else if (sortKey === 'accuracy') {
                aVal = a.genomes[0]?.accuracy ?? 0;
                bVal = b.genomes[0]?.accuracy ?? 0;
            } else if (sortKey === 'fitness') {
                aVal = a.genomes[0]?.adjustedFitness ?? 0;
                bVal = b.genomes[0]?.adjustedFitness ?? 0;
            } else if (sortKey === 'nodes') {
                aVal = a.genomes[0]?.nodes.length ?? 0;
                bVal = b.genomes[0]?.nodes.length ?? 0;
            } else {
                return 0;
            }

            return sortAsc ? aVal - bVal : bVal - aVal;
        });

        return gens;
    }, [generations, sortKey, sortAsc]);

    return (
        <div className={styles.modalOverlay} onClick={onClose}>
            <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                <div className={styles.modalHeader}>
                    <h2 className={styles.modalTitle}>Generations History</h2>
                    <button className={styles.closeBtn} onClick={onClose}>
                        <BsX size={20} />
                    </button>
                </div>

                <div className={styles.controlsBar}>
                    <div className={styles.controlsGroup}>
                        <label htmlFor="sortKey">Sort by:</label>
                        <select
                            id="sortKey"
                            className={styles.sortSelect}
                            value={sortKey}
                            onChange={(e) => setSortKey(e.target.value as SortKey)}
                        >
                            <option value="evalOrder">Eval Order</option>
                            <option value="accuracy">Accuracy</option>
                            <option value="fitness">Fitness</option>
                            <option value="nodes">Nodes</option>
                        </select>
                    </div>
                    <button
                        className={styles.sortDirBtn}
                        onClick={() => setSortAsc((v) => !v)}
                        title={sortAsc ? 'Ascending' : 'Descending'}
                    >
                        {sortAsc ? '↑' : '↓'}
                    </button>
                    <span className={styles.countLabel}>
                        Total: {generations.length} generations
                    </span>
                </div>

                <div className={styles.tableContainer}>
                    <table className={styles.generationsTable}>
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
                                <th>Timestamp</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedGenerations.map((gen) => {
                                const bestFitness = Math.max(...gen.genomes.map((g) => g.adjustedFitness ?? 0));
                                const avgNodes = gen.genomes.reduce((sum, g) => sum + g.nodes.length, 0) / Math.max(1, gen.genomes.length);
                                const peakVram = Math.max(...gen.genomes.map((g) => g.profiler?.peak_active_memory_mb ?? 0));
                                const isSelected = selectedGeneration === gen.generation;

                                return (
                                    <tr
                                        key={gen.generation}
                                        className={`${styles.genRow} ${isSelected ? styles.genRowSelected : ''}`}
                                        onClick={() => onSelectGeneration(gen.generation)}
                                    >
                                        <td className={styles.genNumber}>{gen.generation}</td>
                                        <td>{gen.genomes.length}</td>
                                        <td className={styles.fitValue}>{bestFitness.toFixed(4)}</td>
                                        <td>{avgNodes.toFixed(1)}</td>
                                        <td>
                                            {gen.totalTrainingMs ? (gen.totalTrainingMs / 1000).toFixed(2) : '--'}
                                        </td>
                                        <td>
                                            {gen.totalInferenceMs
                                                ? (gen.totalInferenceMs / Math.max(1, gen.genomes.length)).toFixed(3)
                                                : '--'}
                                        </td>
                                        <td>{peakVram ? peakVram.toFixed(1) : '--'}</td>
                                        <td>{gen.avgSamplesPerSec ? gen.avgSamplesPerSec.toFixed(1) : '--'}</td>
                                        <td className={styles.timestamp}>{gen.timestamp}</td>
                                        <td>
                                            <span className={`${styles.statusBadge} ${gen.evaluated ? styles.statusDone : styles.statusProgress}`}>
                                                {gen.evaluated ? 'Done' : 'Evaluating'}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>

                <div className={styles.modalFooter}>
                    <button className={styles.closeMainBtn} onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};
