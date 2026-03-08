import React from 'react';
import styles from './EvolutionSettingsPanel.module.css';
import { useEvolutionSettingsStore, CrossoverStrategy } from '../../features/evolution-manager/model/store';

interface EvolutionSettingsPanelProps {
    disabled?: boolean;
}

export const EvolutionSettingsPanel: React.FC<EvolutionSettingsPanelProps> = ({ disabled = false }) => {
    const settings = useEvolutionSettingsStore();

    const handleCrossoverChange = (strategy: CrossoverStrategy) => {
        if (disabled) return;
        settings.toggleCrossover(strategy);
    };

    const handleRateChange = (key: keyof typeof settings.mutationRates, e: React.ChangeEvent<HTMLInputElement>) => {
        if (disabled) return;
        settings.setMutationRate(key, parseFloat(e.target.value));
    };

    const crossoverLabels: Record<CrossoverStrategy, string> = {
        'subgraph-insertion': 'Subgraph Insertion',
        'subgraph-replacement': 'Subgraph Replacement',
        'neat-style': 'NEAT Style',
        'multi-point': 'Multi Point',
    };

    const mutationLabels: Record<string, string> = {
        params: 'Parameters',
        addNode: 'Add Node',
        removeNode: 'Remove Node',
        removeSubgraph: 'Remove Subgraph',
        addSkipConnection: 'Add Skip Connection',
        changeLayerType: 'Change Layer Type',
    };

    return (
        <div className={`${styles.panel} ${disabled ? styles.disabled : ''}`}>
            <h3 className={styles.panelTitle}>Evolution Settings</h3>

            {/* Crossover Strategies */}
            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Crossover Strategies</h4>
                <div className={styles.checkboxGroup}>
                    {(['subgraph-insertion', 'subgraph-replacement', 'neat-style', 'multi-point'] as CrossoverStrategy[]).map(strategy => (
                        <label key={strategy} className={styles.checkboxLabel}>
                            <input
                                type="checkbox"
                                checked={settings.selectedCrossovers.includes(strategy)}
                                onChange={() => handleCrossoverChange(strategy)}
                                disabled={disabled}
                            />
                            <span>{crossoverLabels[strategy]}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Mutation Probabilities */}
            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Mutation Probabilities</h4>

                <label className={styles.checkboxLabel}>
                    <input
                        type="checkbox"
                        checked={settings.useAdaptiveMutation}
                        onChange={e => settings.setUseAdaptiveMutation(e.target.checked)}
                        disabled={disabled}
                    />
                    <span>Adaptive Mutation</span>
                </label>

                {settings.useAdaptiveMutation && (
                    <div className={styles.subSetting}>
                        <span className={styles.subLabel}>Target Node Count</span>
                        <input
                            type="number"
                            className={styles.numberInput}
                            value={settings.adaptiveTargetNodes}
                            onChange={e => settings.setAdaptiveTargetNodes(parseInt(e.target.value) || 0)}
                            disabled={disabled}
                        />
                    </div>
                )}

                <div className={styles.sliderGroup}>
                    {Object.entries(settings.mutationRates).map(([key, value]) => {
                        const isAdaptive = settings.useAdaptiveMutation &&
                            (key === 'addNode' || key === 'removeNode' || key === 'removeSubgraph');

                        return (
                            <div key={key} className={styles.sliderRow}>
                                <span className={styles.sliderLabel}>
                                    {mutationLabels[key] || key}
                                    {isAdaptive && <span className={styles.autoTag}>Auto</span>}
                                </span>
                                <div className={styles.sliderControl}>
                                    <input
                                        type="range"
                                        min="0" max="1" step="0.05"
                                        value={value}
                                        onChange={(e) => handleRateChange(key as any, e)}
                                        className={styles.slider}
                                        disabled={disabled || isAdaptive}
                                    />
                                    <span className={styles.sliderValue}>{Math.round(value * 100)}%</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Bloat Control */}
            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Bloat Control</h4>

                <label className={styles.checkboxLabel}>
                    <input
                        type="checkbox"
                        checked={settings.useMaxNodesLimit}
                        onChange={e => settings.setUseMaxNodesLimit(e.target.checked)}
                        disabled={disabled}
                    />
                    <span>Max Nodes Limit</span>
                </label>
                {settings.useMaxNodesLimit && (
                    <div className={styles.subSetting}>
                        <span className={styles.subLabel}>Max Nodes</span>
                        <input
                            type="number"
                            className={styles.numberInput}
                            value={settings.maxNodesLimit}
                            onChange={e => settings.setMaxNodesLimit(parseInt(e.target.value) || 0)}
                            disabled={disabled}
                        />
                    </div>
                )}

                <label className={styles.checkboxLabel}>
                    <input
                        type="checkbox"
                        checked={settings.useParsimonyPressure}
                        onChange={e => settings.setUseParsimonyPressure(e.target.checked)}
                        disabled={disabled}
                    />
                    <span>Parsimony Pressure</span>
                </label>
                {settings.useParsimonyPressure && (
                    <div className={styles.subSetting}>
                        <span className={styles.subLabel}>Alpha (α)</span>
                        <input
                            type="number"
                            step="0.001"
                            className={styles.numberInput}
                            value={settings.parsimonyAlpha}
                            onChange={e => settings.setParsimonyAlpha(parseFloat(e.target.value) || 0)}
                            disabled={disabled}
                        />
                    </div>
                )}
            </div>

            {/* Resource-Aware Fitness */}
            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Resource Awareness</h4>

                <label className={styles.checkboxLabel}>
                    <input
                        type="checkbox"
                        checked={settings.useResourceAwareFitness}
                        onChange={e => settings.setUseResourceAwareFitness(e.target.checked)}
                        disabled={disabled}
                    />
                    <span>Resource-Aware Fitness</span>
                </label>
                {settings.useResourceAwareFitness && (
                    <div className={styles.resourceGroup}>
                        <div className={styles.subSetting}>
                            <span className={styles.subLabel}>Max Flash (bytes)</span>
                            <input
                                type="number"
                                className={styles.numberInput}
                                value={settings.resourceTargets.flash}
                                onChange={e => settings.setResourceTarget('flash', parseInt(e.target.value) || 0)}
                                disabled={disabled}
                            />
                        </div>
                        <div className={styles.subSetting}>
                            <span className={styles.subLabel}>Max RAM (bytes)</span>
                            <input
                                type="number"
                                className={styles.numberInput}
                                value={settings.resourceTargets.ram}
                                onChange={e => settings.setResourceTarget('ram', parseInt(e.target.value) || 0)}
                                disabled={disabled}
                            />
                        </div>
                        <div className={styles.subSetting}>
                            <span className={styles.subLabel}>Max MACs</span>
                            <input
                                type="number"
                                className={styles.numberInput}
                                value={settings.resourceTargets.macs}
                                onChange={e => settings.setResourceTarget('macs', parseInt(e.target.value) || 0)}
                                disabled={disabled}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Training Parameters */}
            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Training Parameters</h4>
                <div className={styles.subSetting}>
                    <span className={styles.subLabel}>Batch Size</span>
                    <input
                        type="number"
                        min="1"
                        max="512"
                        className={styles.numberInput}
                        value={settings.batchSize}
                        onChange={e => settings.setBatchSize(parseInt(e.target.value) || 32)}
                        disabled={disabled}
                    />
                </div>
                <div className={styles.subSetting}>
                    <span className={styles.subLabel}>Epochs per Genome</span>
                    <input
                        type="number"
                        min="1"
                        max="100"
                        className={styles.numberInput}
                        value={settings.evalEpochs}
                        onChange={e => settings.setEvalEpochs(parseInt(e.target.value) || 1)}
                        disabled={disabled}
                    />
                </div>
                <div className={styles.subSetting}>
                    <span className={styles.subLabel}>Dataset Usage</span>
                    <div className={styles.sliderControl}>
                        <input
                            type="range"
                            min="1" max="100" step="1"
                            value={settings.datasetPercent}
                            onChange={e => settings.setDatasetPercent(parseInt(e.target.value))}
                            className={styles.slider}
                            disabled={disabled}
                        />
                        <span className={styles.sliderValue}>{settings.datasetPercent}%</span>
                    </div>
                </div>
                <div className={styles.subSetting}>
                    <span className={styles.subLabel}>Population Size</span>
                    <input
                        type="number"
                        min="4"
                        max="200"
                        className={styles.numberInput}
                        value={settings.populationSize}
                        onChange={e => settings.setPopulationSize(parseInt(e.target.value) || 20)}
                        disabled={disabled}
                    />
                </div>
                <div className={styles.checkboxRow}>
                    <label className={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={settings.useMaxGenerations}
                            onChange={e => settings.setUseMaxGenerations(e.target.checked)}
                            disabled={disabled}
                        />
                        Max Generations
                    </label>
                </div>
                {settings.useMaxGenerations && (
                    <div className={styles.subSetting} style={{ marginLeft: '1.2rem' }}>
                        <span className={styles.subLabel}>Limit</span>
                        <input
                            type="number"
                            min="1"
                            max="10000"
                            className={styles.numberInput}
                            value={settings.maxGenerations}
                            onChange={e => settings.setMaxGenerations(parseInt(e.target.value) || 100)}
                            disabled={disabled}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};
