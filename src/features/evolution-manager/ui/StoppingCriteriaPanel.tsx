import React, { useState } from 'react';
import { useEvolutionSettingsStore } from '../model/store';
import type { StoppingCriterionType } from '../../../shared/lib';
import styles from './StoppingCriteriaPanel.module.css';

function validateSingleCriterion(criterion: StoppingCriterionType): string | null {
    switch (criterion.type) {
        case 'GenerationLimit':
            return criterion.max_generations > 0 ? null : 'Max generations must be greater than 0';
        case 'FitnessPlateau':
            if (criterion.patience_generations <= 0) {
                return 'Patience must be greater than 0';
            }
            if (criterion.improvement_threshold < 0) {
                return 'Improvement threshold cannot be negative';
            }
            return null;
        case 'TimeLimit':
            return criterion.max_seconds > 0 ? null : 'Time limit must be greater than 0 seconds';
        case 'TargetAccuracy':
            return criterion.threshold >= 0 && criterion.threshold <= 1
                ? null
                : 'Target accuracy must be between 0 and 1';
        case 'ManualStop':
            return null;
        default:
            return 'Unknown criterion type';
    }
}

function validateStoppingCriteria(criteria: StoppingCriterionType[], policyType: 'any' | 'all'): string | null {
    if (criteria.length === 0) {
        return 'At least one stopping criterion is required';
    }

    const manualStops = criteria.filter(c => c.type === 'ManualStop');
    if (manualStops.length > 1) {
        return 'Only one ManualStop criterion is allowed';
    }

    for (const criterion of criteria) {
        const error = validateSingleCriterion(criterion);
        if (error) {
            return error;
        }
    }

    if (!['any', 'all'].includes(policyType)) {
        return 'Invalid policy type';
    }

    return null;
}

interface StoppingCriteriaPanelProps {
    disabled?: boolean;
}

type CriterionTypeKey = 'GenerationLimit' | 'FitnessPlateau' | 'TimeLimit' | 'TargetAccuracy' | 'ManualStop';

const CRITERION_TYPES: CriterionTypeKey[] = ['GenerationLimit', 'FitnessPlateau', 'TimeLimit', 'TargetAccuracy', 'ManualStop'];

const CRITERION_LABELS: Record<CriterionTypeKey, string> = {
    GenerationLimit: 'Generation Limit',
    FitnessPlateau: 'Fitness Plateau',
    TimeLimit: 'Time Limit',
    TargetAccuracy: 'Target Accuracy',
    ManualStop: 'Manual Stop',
};

export const StoppingCriteriaPanel: React.FC<StoppingCriteriaPanelProps> = ({ disabled = false }) => {
    const settings = useEvolutionSettingsStore();
    const [addingCriterionType, setAddingCriterionType] = useState<CriterionTypeKey | null>(null);
    const [errors, setErrors] = useState<Record<number, string>>({});

    const handleAddCriterion = (type: CriterionTypeKey) => {
        if (disabled) return;

        let newCriterion: StoppingCriterionType;

        switch (type) {
            case 'GenerationLimit':
                newCriterion = { type: 'GenerationLimit', max_generations: 100 };
                break;
            case 'FitnessPlateau':
                newCriterion = {
                    type: 'FitnessPlateau',
                    patience_generations: 10,
                    improvement_threshold: 0.001,
                    monitor: 'best_fitness',
                };
                break;
            case 'TimeLimit':
                newCriterion = { type: 'TimeLimit', max_seconds: 3600 };
                break;
            case 'TargetAccuracy':
                newCriterion = { type: 'TargetAccuracy', threshold: 0.95 };
                break;
            case 'ManualStop':
                newCriterion = { type: 'ManualStop' };
                break;
        }

        const newCriteria = [...settings.stoppingPolicy.criteria, newCriterion];
        const validationError = validateStoppingCriteria(newCriteria, settings.stoppingPolicy.policy_type);

        if (validationError) {
            setErrors({ ...errors, [newCriteria.length - 1]: validationError });
            return;
        }

        settings.setStoppingPolicy({
            ...settings.stoppingPolicy,
            criteria: newCriteria,
        });

        setAddingCriterionType(null);
        setErrors({});
    };

    const handleRemoveCriterion = (index: number) => {
        if (disabled) return;

        const newCriteria = settings.stoppingPolicy.criteria.filter((_, i) => i !== index);
        const newErrors = { ...errors };
        delete newErrors[index];

        settings.setStoppingPolicy({
            ...settings.stoppingPolicy,
            criteria: newCriteria,
        });

        setErrors(newErrors);
    };

    const handleUpdateCriterion = (index: number, updatedCriterion: StoppingCriterionType) => {
        if (disabled) return;

        const newCriteria = [...settings.stoppingPolicy.criteria];
        newCriteria[index] = updatedCriterion;

        const validationError = validateSingleCriterion(updatedCriterion);
        const newErrors = { ...errors };

        if (validationError) {
            newErrors[index] = validationError;
        } else {
            delete newErrors[index];
        }

        settings.setStoppingPolicy({
            ...settings.stoppingPolicy,
            criteria: newCriteria,
        });

        setErrors(newErrors);
    };

    const handlePolicyTypeChange = (policyType: 'any' | 'all') => {
        if (disabled) return;

        settings.setStoppingPolicy({
            ...settings.stoppingPolicy,
            policy_type: policyType,
        });
    };

    const hasManualStop = settings.stoppingPolicy.criteria.some(c => c.type === 'ManualStop');

    return (
        <div className={`${styles.panel} ${disabled ? styles.disabled : ''}`}>
            <h4 className={styles.panelTitle}>Stopping Criteria</h4>

            {/* Policy Type Selection */}
            <div className={styles.policySection}>
                <span className={styles.policyLabel}>Stop when criteria is:</span>
                <div className={styles.radioGroup}>
                    <label className={styles.radioLabel}>
                        <input
                            type="radio"
                            name="policyType"
                            value="any"
                            checked={settings.stoppingPolicy.policy_type === 'any'}
                            onChange={() => handlePolicyTypeChange('any')}
                            disabled={disabled}
                        />
                        <span>Any met (OR)</span>
                    </label>
                    <label className={styles.radioLabel}>
                        <input
                            type="radio"
                            name="policyType"
                            value="all"
                            checked={settings.stoppingPolicy.policy_type === 'all'}
                            onChange={() => handlePolicyTypeChange('all')}
                            disabled={disabled}
                        />
                        <span>All met (AND)</span>
                    </label>
                </div>
            </div>

            {/* Criteria List */}
            <div className={styles.criteriaList}>
                {settings.stoppingPolicy.criteria.length === 0 ? (
                    <div className={styles.emptyState}>No criteria added. Add at least one.</div>
                ) : (
                    settings.stoppingPolicy.criteria.map((criterion, index) => (
                        <div key={index} className={styles.criterionItem}>
                            <div className={styles.criterionHeader}>
                                <span className={styles.criterionType}>
                                    {CRITERION_LABELS[criterion.type as CriterionTypeKey]}
                                </span>
                                <button
                                    className={styles.removeButton}
                                    onClick={() => handleRemoveCriterion(index)}
                                    disabled={disabled}
                                    title="Remove criterion"
                                >
                                    ✕
                                </button>
                            </div>

                            {errors[index] && <div className={styles.errorMsg}>{errors[index]}</div>}

                            <div className={styles.criterionParams}>
                                {criterion.type === 'GenerationLimit' && (
                                    <div className={styles.paramRow}>
                                        <label>Max Generations:</label>
                                        <input
                                            type="number"
                                            className={styles.numberInput}
                                            value={criterion.max_generations}
                                            onChange={e =>
                                                handleUpdateCriterion(index, {
                                                    ...criterion,
                                                    max_generations: parseInt(e.target.value) || 0,
                                                })
                                            }
                                            min="1"
                                            disabled={disabled}
                                        />
                                    </div>
                                )}

                                {criterion.type === 'FitnessPlateau' && (
                                    <>
                                        <div className={styles.paramRow}>
                                            <label>Monitor:</label>
                                            <select
                                                className={styles.selectInput}
                                                value={criterion.monitor}
                                                onChange={e =>
                                                    handleUpdateCriterion(index, {
                                                        ...criterion,
                                                        monitor: e.target.value as 'best_fitness' | 'pareto_coverage' | 'population_avg',
                                                    })
                                                }
                                                disabled={disabled}
                                            >
                                                <option value="best_fitness">Best Fitness</option>
                                                <option value="pareto_coverage">Pareto Coverage</option>
                                                <option value="population_avg">Population Avg</option>
                                            </select>
                                        </div>
                                        <div className={styles.paramRow}>
                                            <label>Patience (Gen):</label>
                                            <input
                                                type="number"
                                                className={styles.numberInput}
                                                value={criterion.patience_generations}
                                                onChange={e =>
                                                    handleUpdateCriterion(index, {
                                                        ...criterion,
                                                        patience_generations: parseInt(e.target.value) || 0,
                                                    })
                                                }
                                                min="1"
                                                disabled={disabled}
                                            />
                                        </div>
                                        <div className={styles.paramRow}>
                                            <label>Threshold:</label>
                                            <input
                                                type="number"
                                                className={styles.numberInput}
                                                value={criterion.improvement_threshold.toFixed(6)}
                                                onChange={e =>
                                                    handleUpdateCriterion(index, {
                                                        ...criterion,
                                                        improvement_threshold: parseFloat(e.target.value) || 0,
                                                    })
                                                }
                                                min="0"
                                                step="0.001"
                                                disabled={disabled}
                                            />
                                        </div>
                                    </>
                                )}

                                {criterion.type === 'TimeLimit' && (
                                    <div className={styles.paramRow}>
                                        <label>Max Seconds:</label>
                                        <input
                                            type="number"
                                            className={styles.numberInput}
                                            value={criterion.max_seconds}
                                            onChange={e =>
                                                handleUpdateCriterion(index, {
                                                    ...criterion,
                                                    max_seconds: parseInt(e.target.value) || 0,
                                                })
                                            }
                                            min="1"
                                            disabled={disabled}
                                        />
                                    </div>
                                )}

                                {criterion.type === 'TargetAccuracy' && (
                                    <div className={styles.paramRow}>
                                        <label>Threshold (0-1):</label>
                                        <input
                                            type="number"
                                            className={styles.numberInput}
                                            value={criterion.threshold.toFixed(3)}
                                            onChange={e => {
                                                const val = parseFloat(e.target.value) || 0;
                                                handleUpdateCriterion(index, {
                                                    ...criterion,
                                                    threshold: Math.max(0, Math.min(1, val)),
                                                });
                                            }}
                                            min="0"
                                            max="1"
                                            step="0.01"
                                            disabled={disabled}
                                        />
                                    </div>
                                )}

                                {criterion.type === 'ManualStop' && (
                                    <div className={styles.manualStopNote}>
                                        Click "Stop" button during evolution to halt
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Add Criterion */}
            {!addingCriterionType ? (
                <button
                    className={styles.addButton}
                    onClick={() => setAddingCriterionType('GenerationLimit')}
                    disabled={disabled}
                >
                    + Add Criterion
                </button>
            ) : (
                <div className={styles.addCriterionDropdown}>
                    <div className={styles.dropdownHeader}>Select criterion type:</div>
                    <div className={styles.dropdownList}>
                        {CRITERION_TYPES.map(type => (
                            <button
                                key={type}
                                className={styles.dropdownOption}
                                onClick={() => handleAddCriterion(type)}
                                disabled={type === 'ManualStop' && hasManualStop}
                                title={type === 'ManualStop' && hasManualStop ? 'Only one ManualStop allowed' : ''}
                            >
                                {CRITERION_LABELS[type]}
                                {type === 'ManualStop' && hasManualStop && ' (Already added)'}
                            </button>
                        ))}
                    </div>
                    <button
                        className={styles.cancelButton}
                        onClick={() => setAddingCriterionType(null)}
                    >
                        Cancel
                    </button>
                </div>
            )}
        </div>
    );
};
