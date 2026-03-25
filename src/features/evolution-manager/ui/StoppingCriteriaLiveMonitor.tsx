import React, { useEffect, useState } from 'react';
import { useEvolutionSettingsStore } from '../model/store';
import type { StoppingCriterionType } from '../../../shared/lib';
import styles from './StoppingCriteriaLiveMonitor.module.css';

export interface StoppingCriteriaLiveMonitorProps {
    isRunning: boolean;
    generation: number;
    elapsedSeconds: number;
    bestAccuracy: number;
}

export const StoppingCriteriaLiveMonitor: React.FC<StoppingCriteriaLiveMonitorProps> = ({
    isRunning,
    generation,
    elapsedSeconds,
    bestAccuracy,
}) => {
    const settings = useEvolutionSettingsStore();
    const [displayTime, setDisplayTime] = useState('00:00:00');

    // Update timer display
    useEffect(() => {
        const hours = Math.floor(elapsedSeconds / 3600);
        const minutes = Math.floor((elapsedSeconds % 3600) / 60);
        const seconds = elapsedSeconds % 60;
        setDisplayTime(
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
        );
    }, [elapsedSeconds]);

    if (!isRunning) {
        return null;
    }

    const policy = settings.stoppingPolicy;

    return (
        <div className={styles.monitor}>
            <h4 className={styles.monitorTitle}>Stopping Criteria Status</h4>
            <div className={styles.criteriaList}>
                {policy.criteria.map((criterion, index) => (
                    <CriterionMonitor
                        key={index}
                        criterion={criterion}
                        generation={generation}
                        elapsedSeconds={elapsedSeconds}
                        bestAccuracy={bestAccuracy}
                        displayTime={displayTime}
                    />
                ))}
            </div>
            <div className={styles.policyInfo}>
                Stop when: <strong>{policy.policy_type === 'any' ? 'ANY criterion met' : 'ALL criteria met'}</strong>
            </div>
        </div>
    );
};

interface CriterionMonitorProps {
    criterion: StoppingCriterionType;
    generation: number;
    elapsedSeconds: number;
    bestAccuracy: number;
    displayTime: string;
}

const CriterionMonitor: React.FC<CriterionMonitorProps> = ({
    criterion,
    generation,
    elapsedSeconds,
    bestAccuracy,
    displayTime,
}) => {
    switch (criterion.type) {
        case 'GenerationLimit':
            const genProgress = Math.min((generation / criterion.max_generations) * 100, 100);
            return (
                <div className={styles.criterionRow}>
                    <div className={styles.criterionLabel}>Generation Limit</div>
                    <div className={styles.progressContainer}>
                        <div className={styles.progressBar}>
                            <div
                                className={`${styles.progressFill} ${genProgress >= 100 ? styles.complete : ''}`}
                                style={{ width: `${genProgress}%` }}
                            />
                        </div>
                        <span className={styles.progressText}>
                            {generation} / {criterion.max_generations}
                        </span>
                    </div>
                </div>
            );

        case 'TimeLimit':
            const timeProgress = Math.min((elapsedSeconds / criterion.max_seconds) * 100, 100);
            return (
                <div className={styles.criterionRow}>
                    <div className={styles.criterionLabel}>Time Limit</div>
                    <div className={styles.progressContainer}>
                        <div className={styles.progressBar}>
                            <div
                                className={`${styles.progressFill} ${timeProgress >= 100 ? styles.complete : ''}`}
                                style={{ width: `${timeProgress}%` }}
                            />
                        </div>
                        <span className={styles.progressText}>
                            {displayTime} / {formatSeconds(criterion.max_seconds)}
                        </span>
                    </div>
                </div>
            );

        case 'TargetAccuracy':
            const accProgress = Math.min((bestAccuracy / criterion.threshold) * 100, 100);
            return (
                <div className={styles.criterionRow}>
                    <div className={styles.criterionLabel}>Target Accuracy</div>
                    <div className={styles.progressContainer}>
                        <div className={styles.progressBar}>
                            <div
                                className={`${styles.progressFill} ${accProgress >= 100 ? styles.complete : ''}`}
                                style={{ width: `${accProgress}%` }}
                            />
                        </div>
                        <span className={styles.progressText}>
                            {(bestAccuracy * 100).toFixed(1)}% / {(criterion.threshold * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            );

        case 'FitnessPlateau':
            return (
                <div className={styles.criterionRow}>
                    <div className={styles.criterionLabel}>
                        Fitness Plateau ({criterion.monitor})
                    </div>
                    <div className={styles.plateauInfo}>
                        <span className={styles.plateauText}>
                            Patience: {criterion.patience_generations} generations
                        </span>
                    </div>
                </div>
            );

        case 'ManualStop':
            return (
                <div className={styles.criterionRow}>
                    <div className={styles.criterionLabel}>Manual Stop</div>
                    <div className={styles.manualStopNote}>Click Stop button to halt</div>
                </div>
            );

        default:
            return null;
    }
};

function formatSeconds(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}
