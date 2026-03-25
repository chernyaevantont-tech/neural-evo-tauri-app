import React from 'react';
import type { GenomeObjectives } from '../../shared/lib';
import styles from './pareto-front-visualizer.module.css';

type Props = {
    selectedGenome?: GenomeObjectives;
    onUseAsSeed?: (genomeId: string) => void;
    onOpenDetails?: (genomeId: string) => void;
    onExportSelected?: (genomeId: string) => void;
};

export function ParetoSelector({
    selectedGenome,
    onUseAsSeed,
    onOpenDetails,
    onExportSelected,
}: Props) {
    const selectedId = selectedGenome?.genome_id;

    return (
        <div className={styles.selectorCard}>
            <div className={styles.selectorHeader}>Selected Genome</div>
            {selectedGenome ? (
                <div className={styles.selectorMeta} data-testid="pareto-selected-meta">
                    <div>ID: {selectedGenome.genome_id}</div>
                    <div>Accuracy: {selectedGenome.accuracy.toFixed(4)}</div>
                    <div>Latency: {selectedGenome.inference_latency_ms.toFixed(3)} ms</div>
                    <div>Model: {selectedGenome.model_size_mb.toFixed(3)} MB</div>
                    <div>
                        Feasibility:{' '}
                        <span
                            className={
                                selectedGenome.device_feasible === true
                                    ? styles.feasibleBadge
                                    : selectedGenome.device_feasible === false
                                        ? styles.notFeasibleBadge
                                        : styles.unknownBadge
                            }
                        >
                            {selectedGenome.device_feasible === undefined
                                ? 'Unknown'
                                : selectedGenome.device_feasible
                                    ? 'Feasible'
                                    : 'Not feasible'}
                        </span>
                    </div>
                    <div>
                        Constraint score:{' '}
                        {selectedGenome.constraint_violation_score === undefined
                            ? 'n/a'
                            : selectedGenome.constraint_violation_score.toFixed(3)}
                    </div>
                </div>
            ) : (
                <div className={styles.selectorEmpty}>Click any point to inspect actions.</div>
            )}

            <div className={styles.selectorActions}>
                <button
                    type="button"
                    className={styles.actionButton}
                    disabled={!selectedId}
                    onClick={() => selectedId && onUseAsSeed?.(selectedId)}
                >
                    Use as seed
                </button>
                <button
                    type="button"
                    className={styles.actionButton}
                    disabled={!selectedId}
                    onClick={() => selectedId && onOpenDetails?.(selectedId)}
                >
                    Open details
                </button>
                <button
                    type="button"
                    className={styles.actionButton}
                    disabled={!selectedId}
                    onClick={() => selectedId && onExportSelected?.(selectedId)}
                >
                    Export selected
                </button>
            </div>
        </div>
    );
}