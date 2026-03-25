import React from 'react';
import { useEvolutionSettingsStore } from '../../model/store';
import { StoppingCriteriaPanel } from '../StoppingCriteriaPanel';
import styles from './SettingsSections.module.css';

interface StoppingCriteriaSectionProps {
    disabled?: boolean;
}

export function StoppingCriteriaSection({ disabled = false }: StoppingCriteriaSectionProps) {
    const settings = useEvolutionSettingsStore();
    const criteriaCount = settings.stoppingPolicy.criteria.length;

    return (
        <div className={styles.section}>
            <h4 className={styles.sectionTitle}>Stopping Criteria</h4>
            <div className={styles.badges}>
                <span className={styles.badge}>Policy: {settings.stoppingPolicy.policy_type.toUpperCase()}</span>
                <span className={styles.badge}>Active criteria: {criteriaCount}</span>
            </div>
            <StoppingCriteriaPanel disabled={disabled} />
        </div>
    );
}
