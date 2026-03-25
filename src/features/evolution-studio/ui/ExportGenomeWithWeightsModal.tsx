import React, { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Modal } from '../../../shared';
import styles from './ExportGenomeWithWeightsModal.module.css';

type ExportResponse = {
    weights_path: string;
    metadata_path: string;
    used_cached_weights: boolean;
};

type Props = {
    genomeId: string;
    onClose: () => void;
};

export function ExportGenomeWithWeightsModal({ genomeId, onClose }: Props) {
    const [outputPath, setOutputPath] = useState('');
    const [isCheckingCache, setIsCheckingCache] = useState(false);
    const [hasCachedWeights, setHasCachedWeights] = useState<boolean | null>(null);
    const [isExporting, setIsExporting] = useState(false);
    const [status, setStatus] = useState('Ready to export');
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<ExportResponse | null>(null);

    const canExport = useMemo(() => outputPath.trim().length > 0 && !isExporting, [outputPath, isExporting]);

    useEffect(() => {
        let mounted = true;
        const checkCache = async () => {
            setIsCheckingCache(true);
            try {
                const cached = await invoke<boolean>('has_cached_weights', { genomeId });
                if (mounted) {
                    setHasCachedWeights(cached);
                }
            } catch {
                if (mounted) {
                    setHasCachedWeights(null);
                }
            } finally {
                if (mounted) {
                    setIsCheckingCache(false);
                }
            }
        };

        checkCache();
        return () => {
            mounted = false;
        };
    }, [genomeId]);

    const handleBrowse = async () => {
        setError(null);
        const picked = await invoke<string>('pick_folder');
        if (picked) {
            setOutputPath(picked);
        }
    };

    const handleExport = async () => {
        if (!outputPath.trim()) {
            setError('Please select output folder first.');
            return;
        }

        setError(null);
        setResult(null);
        setIsExporting(true);
        setStatus('Exporting mpk weights and metadata...');

        try {
            const response = await invoke<ExportResponse>('export_genome_with_weights', {
                genomeId,
                outputPath,
            });
            setResult(response);
            setStatus(response.used_cached_weights ? 'Export complete (cached weights reused).' : 'Export complete.');
            setHasCachedWeights(true);
        } catch (e) {
            const message = e instanceof Error ? e.message : String(e);
            setError(message || 'Export failed.');
            setStatus('Export failed.');
        } finally {
            setIsExporting(false);
        }
    };

    return (
        <Modal isOpen={true} onClose={onClose} title={`Export Genome ${genomeId}`} maxWidth="640px">
            <div className={styles.container}>
                <div className={styles.section}>
                    <label className={styles.label}>Output folder</label>
                    <div className={styles.pathRow}>
                        <input
                            className={styles.input}
                            value={outputPath}
                            onChange={(e) => setOutputPath(e.target.value)}
                            placeholder="Select target directory"
                            disabled={isExporting}
                        />
                        <button className={styles.buttonSecondary} type="button" onClick={handleBrowse} disabled={isExporting}>
                            Browse
                        </button>
                    </div>
                </div>

                <div className={styles.section}>
                    <div className={styles.cacheRow}>
                        <span className={styles.label}>Cached weights</span>
                        <span className={styles.cacheValue}>
                            {isCheckingCache
                                ? 'Checking...'
                                : hasCachedWeights === null
                                    ? 'Unknown'
                                    : hasCachedWeights
                                        ? 'Available'
                                        : 'Not found'}
                        </span>
                    </div>
                    <div className={styles.status}>{status}</div>
                    {error && <div className={styles.error}>{error}</div>}
                    {result && (
                        <div className={styles.result}>
                            <div>Weights: {result.weights_path}</div>
                            <div>Metadata: {result.metadata_path}</div>
                        </div>
                    )}
                </div>

                <div className={styles.actions}>
                    <button className={styles.buttonSecondary} type="button" onClick={onClose} disabled={isExporting}>
                        Close
                    </button>
                    <button className={styles.buttonPrimary} type="button" onClick={handleExport} disabled={!canExport}>
                        {isExporting ? 'Exporting...' : 'Export'}
                    </button>
                </div>
            </div>
        </Modal>
    );
}
