import React from 'react';
import type { PopulatedGenome } from '../../../entities/genome';
import styles from './GenomeDetailPanel.module.css';

interface GenomeDetailPanelProps {
    genome: PopulatedGenome;
    onOpenProfiler: (genome: PopulatedGenome) => void;
}

export function GenomeDetailPanel({ genome, onOpenProfiler }: GenomeDetailPanelProps) {
    if (!genome.profiler) {
        return null;
    }

    return (
        <div className={styles.panel}>
            <div className={styles.meta}>
                Training: {(genome.profiler.total_train_duration_ms / 1000).toFixed(2)}s | Inference:{' '}
                {genome.profiler.inference_msec_per_sample.toFixed(3)}ms | Peak:{' '}
                {genome.profiler.peak_active_memory_mb.toFixed(1)}MB
            </div>
            <button className={styles.button} onClick={() => onOpenProfiler(genome)}>
                View Profiler Details
            </button>
        </div>
    );
}
