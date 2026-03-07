import React, { useEffect, useState } from 'react';
import styles from './DatasetManagerPage.module.css';
import { invoke } from '@tauri-apps/api/core';

interface Props {
    rootPath: string;
    indexPath: string;
    hasHeaders: boolean;
    onClose: () => void;
}

interface CsvPreview {
    headers: string[];
    rows: string[][];
}

export const CsvPreviewModal: React.FC<Props> = ({ rootPath, indexPath, hasHeaders, onClose }) => {
    const [preview, setPreview] = useState<CsvPreview | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchPreview = async () => {
            try {
                const data = await invoke<CsvPreview>('preview_csv', {
                    rootPath,
                    indexPath,
                    hasHeaders,
                    rows: 10
                });
                setPreview(data);
            } catch (err: any) {
                setError(err.toString());
            } finally {
                setIsLoading(false);
            }
        };

        if (rootPath && indexPath) {
            fetchPreview();
        } else {
            setError("Invalid Root Directory or CSV Path.");
            setIsLoading(false);
        }
    }, [rootPath, indexPath, hasHeaders]);

    return (
        <div className={styles.modalOverlay}>
            <div className={styles.modalContent} style={{ maxWidth: '800px', width: '90%' }}>
                <h2>CSV Preview: {indexPath}</h2>
                <div className={styles.formGroup} style={{ maxHeight: '400px', overflow: 'auto', background: 'var(--color-bg-primary)', padding: '1rem', borderRadius: '4px' }}>
                    {isLoading && <p>Loading preview...</p>}
                    {error && <p style={{ color: 'var(--color-danger)' }}>{error}</p>}
                    {preview && (
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead>
                                <tr>
                                    {preview.headers.map((h, i) => (
                                        <th key={i} style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '2px solid var(--color-border)', color: 'var(--color-text-secondary)' }}>
                                            {hasHeaders ? h : `${i} (${h})`}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {preview.rows.length === 0 && (
                                    <tr><td colSpan={preview.headers.length} style={{ padding: '0.5rem' }}>No rows found.</td></tr>
                                )}
                                {preview.rows.map((row, rIdx) => (
                                    <tr key={rIdx} style={{ borderBottom: '1px solid var(--color-border)' }}>
                                        {row.map((cell, cIdx) => (
                                            <td key={cIdx} style={{ padding: '0.5rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '150px' }}>
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
                <div className={styles.modalActions}>
                    <button className={styles.cancelBtn} onClick={onClose}>Close</button>
                </div>
            </div>
        </div>
    );
};
