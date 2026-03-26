import React, { useState } from 'react';
import { useEvolutionSettingsStore } from '../../model/store';
import styles from './SettingsSections.module.css';

interface AdvancedPerformanceSectionProps {
    disabled?: boolean;
}

export function AdvancedPerformanceSection({ disabled = false }: AdvancedPerformanceSectionProps) {
    const settings = useEvolutionSettingsStore();
    const [expanded, setExpanded] = useState(false);

    return (
        <div className={styles.section}>
            <h4 className={styles.sectionTitle}>Advanced Performance</h4>
            <button
                type="button"
                className={styles.select}
                onClick={() => setExpanded((prev) => !prev)}
                disabled={disabled}
            >
                {expanded ? 'Hide advanced settings' : 'Show advanced settings'}
            </button>

            {expanded && (
                <>
                    <label className={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={settings.profilingEnabled}
                            onChange={(event) => settings.setProfilingEnabled(event.target.checked)}
                            disabled={disabled}
                        />
                        <span>Profiling enabled</span>
                    </label>

                    <div className={styles.row}>
                        <span className={styles.label}>Safety margin (MB)</span>
                        <input
                            className={styles.numberInput}
                            type="number"
                            min="0"
                            step="1"
                            value={settings.memorySafetyMarginMb}
                            onChange={(event) => settings.setMemorySafetyMarginMb(Number(event.target.value) || 0)}
                            disabled={disabled}
                        />
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Estimator safety factor</span>
                        <input
                            className={styles.numberInput}
                            type="number"
                            min="1"
                            step="0.05"
                            value={settings.estimatorSafetyFactor}
                            onChange={(event) => settings.setEstimatorSafetyFactor(Number(event.target.value) || 1)}
                            disabled={disabled}
                        />
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Memory mode</span>
                        <select
                            className={styles.select}
                            value={settings.memoryMode}
                            onChange={(event) => settings.setMemoryMode(event.target.value as 'estimate' | 'runtime' | 'hybrid')}
                            disabled={disabled}
                        >
                            <option value="estimate">estimate</option>
                            <option value="runtime">runtime</option>
                            <option value="hybrid">hybrid</option>
                        </select>
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Execution mode</span>
                        <select
                            className={styles.select}
                            value={settings.executionMode}
                            onChange={(event) => settings.setExecutionMode(event.target.value as 'sequential' | 'parallel-cpu' | 'parallel-safe-limited')}
                            disabled={disabled}
                        >
                            <option value="sequential">sequential</option>
                            <option value="parallel-safe-limited">parallel-safe-limited</option>
                            <option value="parallel-cpu">parallel-cpu</option>
                        </select>
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Max parallel jobs</span>
                        <input
                            className={styles.numberInput}
                            type="number"
                            min="1"
                            max="64"
                            step="1"
                            value={settings.maxParallelJobs}
                            onChange={(event) => settings.setMaxParallelJobs(Number(event.target.value) || 1)}
                            disabled={disabled || settings.executionMode === 'sequential'}
                        />
                    </div>
                </>
            )}
        </div>
    );
}
