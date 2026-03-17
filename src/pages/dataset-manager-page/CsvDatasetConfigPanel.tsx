import React, { useState } from 'react';
import styles from './DatasetManagerPage.module.css';
import { CsvPreprocessingConfig } from '../../features/dataset-manager/model/store';
import { BsEye } from 'react-icons/bs';

interface CsvDatasetLocator {
    type: 'CsvDataset';
    csvPath: string;
    hasHeaders: boolean;
    sampleMode: 'row' | 'temporal_window';
    featureColumns: string[];
    targetColumn: string;
    windowSize?: number;
    windowStride?: number;
    preprocessing: CsvPreprocessingConfig;
}

interface Props {
    locator: CsvDatasetLocator;
    onChange: (locator: CsvDatasetLocator) => void;
    onPreview?: () => void;
    previewDisabled?: boolean;
    role?: 'Input' | 'Target' | 'Ignore';  // ← Add role information
}

export const CsvDatasetConfigPanel: React.FC<Props> = ({ 
    locator, 
    onChange, 
    onPreview,
    previewDisabled,
    role
}) => {
    // For Target streams, force featureColumns to be empty
    const isTargetStream = role === 'Target';
    const effectiveColumns = isTargetStream ? [] : locator.featureColumns;
    
    const [columnInput, setColumnInput] = useState<string>(
        !isTargetStream && locator.featureColumns.length === 1 && locator.featureColumns[0].includes(':')
            ? locator.featureColumns[0]
            : effectiveColumns.join(', ')
    );

    const parseColumnRange = (input: string): string[] => {
        // Parse "ch0:ch11" or "0:11" or "col1, col2, col3" into array
        const trimmed = input.trim();
        
        // Try format: "prefix+startNum:prefix+endNum" (e.g., "ch0:ch11")
        const prefixRangeMatch = trimmed.match(/^(\D+?)(\d+):(\D*?)(\d+)$/);
        if (prefixRangeMatch) {
            const [, prefix1, startStr, prefix2, endStr] = prefixRangeMatch;
            // Both parts should have same prefix
            if (prefix1 === prefix2) {
                const start = parseInt(startStr);
                const end = parseInt(endStr);
                const result = [];
                for (let i = start; i <= end; i++) {
                    result.push(`${prefix1}${i}`);
                }
                return result;
            }
        }

        // Try simple numeric range: "0:11"
        const numRangeMatch = trimmed.match(/^(\d+):(\d+)$/);
        if (numRangeMatch) {
            const [, start, end] = numRangeMatch;
            const result = [];
            for (let i = parseInt(start); i <= parseInt(end); i++) {
                result.push(i.toString());
            }
            return result;
        }

        // Fallback: comma-separated list
        return trimmed.split(',').map(s => s.trim()).filter(s => s.length > 0);
    };

    const handleColumnInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        setColumnInput(value);
        const parsed = parseColumnRange(value);
        if (parsed.length > 0) {
            onChange({ ...locator, featureColumns: parsed });
        }
    };

    return (
        <div style={{ borderTop: '1px dashed var(--color-border)', paddingTop: '0.75rem' }}>
            <div style={{ marginBottom: '1rem' }}>
                <h4 style={{ margin: '0 0 0.75rem 0', color: 'var(--color-text-secondary)' }}>
                    CSV Dataset Configuration
                </h4>
                <p style={{ margin: '0 0 0.75rem 0', color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
                    Configure how to load data from your CSV file
                </p>
            </div>

            {/* CSV Path */}
            <div className={styles.configRow}>
                <div className={styles.inputGroup} style={{ flex: 1 }}>
                    <label>CSV File Path (Relative to Root)</label>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <input
                            type="text"
                            value={locator.csvPath}
                            onChange={(e) => onChange({ ...locator, csvPath: e.target.value })}
                            placeholder="e.g. data.csv"
                            style={{ flex: 1 }}
                        />
                        <button
                            onClick={onPreview}
                            disabled={previewDisabled}
                            style={{
                                background: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border)',
                                color: 'var(--color-text-primary)',
                                padding: '0.4rem 0.8rem',
                                borderRadius: '4px',
                                cursor: previewDisabled ? 'not-allowed' : 'pointer',
                                opacity: previewDisabled ? 0.5 : 1,
                                display: 'flex',
                                gap: '0.4rem',
                                alignItems: 'center',
                                whiteSpace: 'nowrap'
                            }}
                        >
                            <BsEye /> Preview
                        </button>
                    </div>
                </div>
                <div className={`${styles.inputGroup} ${styles.checkboxRow}`}>
                    <input
                        type="checkbox"
                        checked={locator.hasHeaders}
                        onChange={(e) => onChange({ ...locator, hasHeaders: e.target.checked })}
                    />
                    <label>CSV has Headers</label>
                </div>
            </div>

            {/* Sample Mode Selection */}
            <div className={styles.configRow}>
                <div style={{ flex: 1 }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem' }}>Sample Mode</label>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                            <input
                                type="radio"
                                name={`sampleMode-${locator.csvPath}`}
                                checked={locator.sampleMode === 'row'}
                                onChange={() => onChange({ ...locator, sampleMode: 'row' })}
                            />
                            Row Mode (each CSV row = 1 sample)
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                            <input
                                type="radio"
                                name={`sampleMode-${locator.csvPath}`}
                                checked={locator.sampleMode === 'temporal_window'}
                                onChange={() => onChange({ ...locator, sampleMode: 'temporal_window' })}
                            />
                            Temporal Window (sliding window = 1 sample)
                        </label>
                    </div>
                </div>
            </div>

            {/* Temporal Window Settings (only if temporal_window mode) */}
            {locator.sampleMode === 'temporal_window' && (
                <div className={styles.configRow}>
                    <div className={styles.inputGroup}>
                        <label>Window Size (timesteps)</label>
                        <input
                            type="number"
                            min="1"
                            value={locator.windowSize || 50}
                            onChange={(e) => onChange({ ...locator, windowSize: parseInt(e.target.value) })}
                            style={{ width: '100px' }}
                        />
                    </div>
                    <div className={styles.inputGroup}>
                        <label>Window Stride</label>
                        <input
                            type="number"
                            min="1"
                            value={locator.windowStride || 1}
                            onChange={(e) => onChange({ ...locator, windowStride: parseInt(e.target.value) })}
                            style={{ width: '100px' }}
                        />
                    </div>
                    <div className={styles.inputGroup} style={{ flex: 1 }}>
                        <small style={{ color: 'var(--color-text-muted)', display: 'block' }}>
                            Creates overlapping samples. Stride=1 for max overlap, larger for sparser samples.
                        </small>
                    </div>
                </div>
            )}

            {/* Feature Columns - Only for Input streams */}
            {!isTargetStream ? (
                <>
                    <div className={styles.inputGroup}>
                        <label>Feature Columns</label>
                        <input
                            type="text"
                            value={columnInput}
                            onChange={handleColumnInputChange}
                            placeholder='e.g. "ch0:ch11" or "col1, col2, col3" or "0:11" (for indices)'
                        />
                        <small style={{ color: 'var(--color-text-muted)', display: 'block', marginTop: '0.25rem' }}>
                            Use range notation (ch0:ch11) or comma-separated list. Selected: {locator.featureColumns.length} columns
                        </small>
                    </div>

                    {/* Feature Columns Preview */}
                    {locator.featureColumns.length > 0 && (
                        <div style={{ 
                            background: 'var(--color-bg-secondary)', 
                            border: '1px solid var(--color-border)', 
                            borderRadius: '4px',
                            padding: '0.75rem',
                            marginBottom: '1rem',
                            fontSize: '0.85rem'
                        }}>
                            <strong style={{ color: 'var(--color-text-secondary)' }}>Columns:</strong> {locator.featureColumns.join(', ')}
                            <br />
                            <strong style={{ color: 'var(--color-text-secondary)' }}>Shape inference:</strong> 
                            {locator.sampleMode === 'row' 
                                ? ` [${locator.featureColumns.length}]` 
                                : ` [${locator.windowSize || 50}, ${locator.featureColumns.length}]`
                            }
                        </div>
                    )}
                </>
            ) : (
                <div style={{ 
                    background: 'var(--color-bg-secondary)', 
                    border: '1px solid var(--color-border-warning)', 
                    borderRadius: '4px',
                    padding: '0.75rem',
                    marginBottom: '1rem',
                    fontSize: '0.85rem',
                    color: 'var(--color-text-secondary)'
                }}>
                    <strong>Feature Columns: N/A</strong><br />
                    <small>Target streams use only the Target Column for labels. No feature selection needed.</small>
                </div>
            )}

            {/* Target Column - Only for Target streams */}
            {isTargetStream ? (
                <div className={styles.inputGroup} style={{ marginBottom: '1rem' }}>
                    <label>Target Column (Label/Class)</label>
                    <input
                        type="text"
                        value={locator.targetColumn}
                        onChange={(e) => onChange({ ...locator, targetColumn: e.target.value })}
                        placeholder={locator.hasHeaders ? 'e.g. gesture, class, label' : 'e.g. column index'}
                    />
                    <small style={{ color: 'var(--color-text-muted)', display: 'block', marginTop: '0.25rem' }}>
                        Column containing class labels for classification tasks
                    </small>
                </div>
            ) : (
                <div style={{ 
                    background: 'var(--color-bg-secondary)', 
                    border: '1px solid var(--color-border-warning)', 
                    borderRadius: '4px',
                    padding: '0.75rem',
                    marginBottom: '1rem',
                    fontSize: '0.85rem',
                    color: 'var(--color-text-secondary)'
                }}>
                    <strong>Target Column: N/A</strong><br />
                    <small>Input streams load only Feature Columns. Targets are handled by Target streams.</small>
                </div>
            )}

            {/* Preprocessing */}
            <div style={{ borderTop: '1px dashed var(--color-border)', paddingTop: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                    <strong style={{ color: 'var(--color-text-secondary)' }}>Preprocessing</strong>
                </div>

                <div className={styles.configRow}>
                    <div className={styles.inputGroup}>
                        <label>Normalization</label>
                        <select
                            value={locator.preprocessing.normalization}
                            onChange={(e) => onChange({
                                ...locator,
                                preprocessing: { ...locator.preprocessing, normalization: e.target.value as any }
                            })}
                        >
                            <option value="none">None (raw values)</option>
                            <option value="global">Global (single mean/std)</option>
                            <option value="per-sample">Per-Sample (normalize within each sample)</option>
                            <option value="per-channel">Per-Channel (each channel independently)</option>
                        </select>
                        <small style={{ color: 'var(--color-text-muted)', marginTop: '0.25rem', display: 'block' }}>
                            {locator.preprocessing.normalization === 'per-channel' && 'Z-score normalize each channel. Best for multi-channel temporal data.'}
                            {locator.preprocessing.normalization === 'per-sample' && 'Normalize entire sample together. Useful for amplitude-invariant patterns.'}
                            {locator.preprocessing.normalization === 'global' && 'Single statistics for all data. Use when all samples share similar scale.'}
                            {locator.preprocessing.normalization === 'none' && 'Use raw values as-is.'}
                        </small>
                    </div>

                    <div className={styles.inputGroup}>
                        <label>Missing Values</label>
                        <select
                            value={locator.preprocessing.handleMissing}
                            onChange={(e) => onChange({
                                ...locator,
                                preprocessing: { ...locator.preprocessing, handleMissing: e.target.value as any }
                            })}
                        >
                            <option value="skip">Skip rows with NaN</option>
                            <option value="interpolate">Linear interpolation</option>
                            <option value="mean">Fill with column mean</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    );
};
