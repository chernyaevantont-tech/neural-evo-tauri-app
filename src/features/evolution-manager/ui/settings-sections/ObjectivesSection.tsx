import React, { useMemo } from 'react';
import { useEvolutionSettingsStore, type SecondaryObjective } from '../../model/store';
import {
    normalizeObjectiveWeights,
    validateObjectives,
} from '../../model/objectives';
import styles from './SettingsSections.module.css';

interface ObjectivesSectionProps {
    disabled?: boolean;
}

const AVAILABLE_OBJECTIVES: Array<{ key: SecondaryObjective; label: string }> = [
    { key: 'latency', label: 'Latency' },
    { key: 'model_size', label: 'Model size' },
    { key: 'train_time', label: 'Train time' },
];

export function ObjectivesSection({ disabled = false }: ObjectivesSectionProps) {
    const settings = useEvolutionSettingsStore();

    const validationError = useMemo(() => {
        return validateObjectives(settings.mobjEnabled ? 'multi' : 'single', settings.secondaryObjectives);
    }, [settings.mobjEnabled, settings.secondaryObjectives]);

    const normalizedSum = useMemo(() => {
        const normalized = normalizeObjectiveWeights(settings.objectiveWeights, settings.secondaryObjectives);
        return Object.values(normalized).reduce((acc, value) => acc + value, 0);
    }, [settings.objectiveWeights, settings.secondaryObjectives]);

    const toggleSecondaryObjective = (value: SecondaryObjective) => {
        if (disabled) {
            return;
        }

        const exists = settings.secondaryObjectives.includes(value);
        const next = exists
            ? settings.secondaryObjectives.filter((objective) => objective !== value)
            : [...settings.secondaryObjectives, value];

        settings.setSecondaryObjectives(next);
    };

    return (
        <div className={styles.section}>
            <h4 className={styles.sectionTitle}>Objectives</h4>

            <div className={styles.row}>
                <span className={styles.label}>Optimization mode</span>
                <select
                    className={styles.select}
                    value={settings.mobjEnabled ? 'multi' : 'single'}
                    onChange={(event) => settings.setMobjEnabled(event.target.value === 'multi')}
                    disabled={disabled}
                >
                    <option value="single">Single-Objective</option>
                    <option value="multi">Multi-Objective (Pareto)</option>
                </select>
            </div>

            {settings.mobjEnabled && (
                <>
                    <div className={styles.checkboxGroup}>
                        {AVAILABLE_OBJECTIVES.map((objective) => (
                            <label key={objective.key} className={styles.checkboxLabel}>
                                <input
                                    type="checkbox"
                                    checked={settings.secondaryObjectives.includes(objective.key)}
                                    onChange={() => toggleSecondaryObjective(objective.key)}
                                    disabled={disabled}
                                />
                                <span>{objective.label}</span>
                            </label>
                        ))}
                    </div>

                    {validationError && <div className={styles.error}>{validationError}</div>}

                    <label className={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={settings.objectiveWeightsEnabled}
                            onChange={(event) => settings.setObjectiveWeightsEnabled(event.target.checked)}
                            disabled={disabled}
                        />
                        <span>Use weighted aggregation</span>
                    </label>

                    {settings.objectiveWeightsEnabled && (
                        <>
                            <div className={styles.sliderRow}>
                                <span className={styles.label}>Accuracy weight</span>
                                <div className={styles.sliderTrack}>
                                    <input
                                        className={styles.slider}
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={settings.objectiveWeights.accuracy}
                                        onChange={(event) => settings.setObjectiveWeight('accuracy', Number(event.target.value))}
                                        disabled={disabled}
                                    />
                                    <span className={styles.sliderValue}>{settings.objectiveWeights.accuracy.toFixed(2)}</span>
                                </div>
                            </div>
                            <div className={styles.sliderRow}>
                                <span className={styles.label}>Latency weight</span>
                                <div className={styles.sliderTrack}>
                                    <input
                                        className={styles.slider}
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={settings.objectiveWeights.latency}
                                        onChange={(event) => settings.setObjectiveWeight('latency', Number(event.target.value))}
                                        disabled={disabled}
                                    />
                                    <span className={styles.sliderValue}>{settings.objectiveWeights.latency.toFixed(2)}</span>
                                </div>
                            </div>
                            <div className={styles.sliderRow}>
                                <span className={styles.label}>Model size weight</span>
                                <div className={styles.sliderTrack}>
                                    <input
                                        className={styles.slider}
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={settings.objectiveWeights.model_size}
                                        onChange={(event) => settings.setObjectiveWeight('model_size', Number(event.target.value))}
                                        disabled={disabled}
                                    />
                                    <span className={styles.sliderValue}>{settings.objectiveWeights.model_size.toFixed(2)}</span>
                                </div>
                            </div>
                            <div className={styles.sliderRow}>
                                <span className={styles.label}>Train time weight</span>
                                <div className={styles.sliderTrack}>
                                    <input
                                        className={styles.slider}
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={settings.objectiveWeights.train_time}
                                        onChange={(event) => settings.setObjectiveWeight('train_time', Number(event.target.value))}
                                        disabled={disabled}
                                    />
                                    <span className={styles.sliderValue}>{settings.objectiveWeights.train_time.toFixed(2)}</span>
                                </div>
                            </div>
                            <button
                                type="button"
                                className={styles.select}
                                onClick={() => settings.normalizeObjectiveWeights()}
                                disabled={disabled}
                            >
                                Normalize weights
                            </button>
                            <p className={styles.helper}>Weight sum: {normalizedSum.toFixed(2)} (normalized target is 1.00)</p>
                        </>
                    )}
                </>
            )}
        </div>
    );
}
