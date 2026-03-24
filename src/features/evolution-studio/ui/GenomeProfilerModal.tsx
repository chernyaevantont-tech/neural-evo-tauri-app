import React from 'react';
import type { TrainingProfiler } from '../../../shared/lib';
import styles from './GenomeProfilerModal.module.css';

interface GenomeProfilerModalProps {
    genomeId: string;
    profiler: TrainingProfiler;
    onClose: () => void;
}

function formatSecondsFromMs(ms: number): string {
    return `${(ms / 1000).toFixed(2)}s`;
}

function formatMs(ms: number): string {
    return `${ms.toFixed(3)}ms`;
}

function formatMb(mb: number): string {
    return `${mb.toFixed(1)}MB`;
}

export function GenomeProfilerModal({ genomeId, profiler, onClose }: GenomeProfilerModalProps) {
    return (
        <div className={styles.overlay} onClick={onClose}>
            <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
                <div className={styles.header}>
                    <h2 className={styles.title}>Profiler: {genomeId}</h2>
                    <button className={styles.closeButton} onClick={onClose}>
                        Close
                    </button>
                </div>

                <div className={styles.grid}>
                    <section className={styles.card}>
                        <h3>Training</h3>
                        <dl>
                            <dt>Duration</dt>
                            <dd>{formatSecondsFromMs(profiler.total_train_duration_ms)}</dd>
                            <dt>First Batch</dt>
                            <dd>{formatSecondsFromMs(profiler.first_batch_ms)}</dd>
                            <dt>Batch Count</dt>
                            <dd>{profiler.batch_count}</dd>
                            <dt>Throughput</dt>
                            <dd>{profiler.samples_per_sec.toFixed(1)} samples/sec</dd>
                            <dt>Early Stop Epoch</dt>
                            <dd>{profiler.early_stop_epoch ?? 'N/A'}</dd>
                        </dl>
                    </section>

                    <section className={styles.card}>
                        <h3>Validation & Test</h3>
                        <dl>
                            <dt>Validation Duration</dt>
                            <dd>{formatSecondsFromMs(profiler.val_duration_ms)}</dd>
                            <dt>Inference / Sample</dt>
                            <dd>{formatMs(profiler.inference_msec_per_sample)}</dd>
                            <dt>Test Duration</dt>
                            <dd>{formatSecondsFromMs(profiler.test_duration_ms)}</dd>
                        </dl>
                    </section>

                    <section className={styles.card}>
                        <h3>Memory Peaks</h3>
                        <dl>
                            <dt>Total Active</dt>
                            <dd>{formatMb(profiler.peak_active_memory_mb)}</dd>
                            <dt>Model Params</dt>
                            <dd>{formatMb(profiler.peak_model_params_mb)}</dd>
                            <dt>Gradients</dt>
                            <dd>{formatMb(profiler.peak_gradient_mb)}</dd>
                            <dt>Optimizer State</dt>
                            <dd>{formatMb(profiler.peak_optim_state_mb)}</dd>
                            <dt>Activations</dt>
                            <dd>{formatMb(profiler.peak_activation_mb)}</dd>
                        </dl>
                    </section>
                </div>
            </div>
        </div>
    );
}
