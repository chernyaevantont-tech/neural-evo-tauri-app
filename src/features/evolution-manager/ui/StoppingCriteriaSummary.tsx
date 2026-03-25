import React from 'react';
import styles from './StoppingCriteriaSummary.module.css';
import { getCriterionDescription } from '../model/stoppingCriteriaValidator';
import type { StoppingCriterionType } from '../../../shared/lib';

export interface StoppingCriteriaSummaryProps {
    /**
     * Which criterion caused the stop (by index)
     */
    triggeredCriterionIndex: number | null;
    /**
     * All criteria that were defined
     */
    criteria: StoppingCriterionType[];
    /**
     * Final generation count
     */
    finalGeneration: number;
    /**
     * Total elapsed time in seconds
     */
    elapsedSeconds: number;
    /**
     * Final best accuracy achieved
     */
    finalAccuracy: number;
}

export const StoppingCriteriaSummary: React.FC<StoppingCriteriaSummaryProps> = ({
    triggeredCriterionIndex,
    criteria,
    finalGeneration,
    elapsedSeconds,
    finalAccuracy,
}) => {
    const formatTime = (seconds: number) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    };

    const triggeredCriterion =
        triggeredCriterionIndex !== null ? criteria[triggeredCriterionIndex] : null;

    return (
        <div className={styles.summary}>
            <h4 className={styles.summaryTitle}>Evolution Stopped</h4>

            {triggeredCriterion && (
                <div className={styles.triggeredSection}>
                    <div className={styles.sectionLabel}>Triggered by</div>
                    <div className={styles.triggeredCriterion}>
                        <span className={styles.criterionType}>{triggeredCriterion.type}</span>
                        <span className={styles.criterionDesc}>
                            {getCriterionDescription(triggeredCriterion)}
                        </span>
                    </div>
                </div>
            )}

            <div className={styles.statsSection}>
                <div className={styles.sectionLabel}>Statistics</div>
                <div className={styles.statsList}>
                    <div className={styles.statItem}>
                        <span className={styles.statLabel}>Final Generation</span>
                        <span className={styles.statValue}>{finalGeneration}</span>
                    </div>
                    <div className={styles.statItem}>
                        <span className={styles.statLabel}>Elapsed Time</span>
                        <span className={styles.statValue}>{formatTime(elapsedSeconds)}</span>
                    </div>
                    <div className={styles.statItem}>
                        <span className={styles.statLabel}>Best Accuracy</span>
                        <span className={styles.statValue}>{(finalAccuracy * 100).toFixed(2)}%</span>
                    </div>
                </div>
            </div>

            {criteria.length > 1 && (
                <div className={styles.allCriteriaSection}>
                    <div className={styles.sectionLabel}>All Criteria</div>
                    <div className={styles.criteriaList}>
                        {criteria.map((criterion, index) => (
                            <div
                                key={index}
                                className={`${styles.criteriaItem} ${
                                    index === triggeredCriterionIndex ? styles.triggered : ''
                                }`}
                            >
                                {index === triggeredCriterionIndex && (
                                    <span className={styles.checkmark}>✓</span>
                                )}
                                <span className={styles.criteriaText}>
                                    {getCriterionDescription(criterion)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
