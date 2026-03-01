import React from 'react';
import styles from './DatasetManagerPage.module.css';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import { BsDatabaseAdd, BsFolder2Open, BsFiletypeCsv, BsSearch, BsCheckCircle, BsExclamationTriangle } from 'react-icons/bs';
import { useDatasetManagerStore, DatasetSourceType, ScanResult } from '../../features/dataset-manager/model/store';
import { CreateDatasetModal } from './CreateDatasetModal';
import { DataStreamsPanel } from './DataStreamsPanel';
import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

export const DatasetManagerPage: React.FC = () => {
    const { profiles, selectedProfileId, setSelectedProfileId, removeProfile } = useDatasetManagerStore();
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isScanning, setIsScanning] = useState(false);

    const getIcon = (type: DatasetSourceType) => {
        switch (type) {
            case "CSV": return <BsFiletypeCsv />;
            case "Folder":
            default: return <BsFolder2Open />;
        }
    }

    const handleScan = async (profileId: string) => {
        const profile = useDatasetManagerStore.getState().profiles.find(p => p.id === profileId);
        if (!profile || !profile.sourcePath || profile.streams.length === 0) return;

        setIsScanning(true);
        try {
            const streamConfigs = profile.streams.map(s => ({
                stream_id: s.id,
                alias: s.alias,
                locator_type: s.locator.type,
                pattern: s.locator.type === 'GlobPattern' ? s.locator.pattern : null,
                path_template: s.locator.type === 'CompanionFile' ? s.locator.pathTemplate : null,
            }));

            const result = await invoke<{
                total_matched: number;
                dropped_count: number;
                stream_reports: Array<{
                    stream_id: string;
                    alias: string;
                    found_count: number;
                    missing_sample_ids: string[];
                    discovered_classes: Record<string, number> | null;
                }>;
            }>('scan_dataset', {
                rootPath: profile.sourcePath,
                streamConfigs,
            });

            const scanResult: ScanResult = {
                totalMatched: result.total_matched,
                droppedCount: result.dropped_count,
                streamReports: result.stream_reports.map(r => ({
                    streamId: r.stream_id,
                    alias: r.alias,
                    foundCount: r.found_count,
                    missingSampleIds: r.missing_sample_ids,
                    discoveredClasses: r.discovered_classes ?? undefined,
                })),
                timestamp: new Date().toISOString(),
            };

            useDatasetManagerStore.getState().updateProfile(profileId, {
                scanResult,
                totalSamples: scanResult.totalMatched,
                isScanned: true,
            });
        } catch (err) {
            console.error('Scan failed:', err);
        } finally {
            setIsScanning(false);
        }
    };

    return (
        <>
            <TitleBar />
            <div className={styles.container}>
                <div className={styles.content}>
                    {/* Left Sidebar: Dataset Profiles List */}
                    <div className={styles.sidebar}>
                        <div className={styles.sidebarHeader}>
                            <h2 className={styles.sidebarTitle}>Dataset Profiles</h2>
                            <button className={styles.addButton} title="Create New Profile" onClick={() => setIsCreateModalOpen(true)}>
                                +
                            </button>
                        </div>
                        <div className={styles.datasetList}>
                            {profiles.map(p => (
                                <div
                                    key={p.id}
                                    className={`${styles.datasetItem} ${selectedProfileId === p.id ? styles.active : ''}`}
                                    onClick={() => setSelectedProfileId(p.id)}
                                >
                                    <div className={styles.datasetIcon}>{getIcon(p.type)}</div>
                                    <div className={styles.datasetInfo}>
                                        <span className={styles.datasetName}>{p.name}</span>
                                        <span className={styles.datasetType}>
                                            {p.type} · {p.streams.filter(s => s.role === 'Input').length} in / {p.streams.filter(s => s.role === 'Target').length} out
                                            {p.isScanned && ` · ${p.totalSamples} samples`}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right Main Panel: Configuration & Preview */}
                    <div className={styles.mainPanel}>
                        {selectedProfileId && profiles.find(p => p.id === selectedProfileId) ? (() => {
                            const profile = profiles.find(p => p.id === selectedProfileId)!;

                            const handleUpdate = (updates: Partial<typeof profile>) => {
                                useDatasetManagerStore.getState().updateProfile(profile.id, updates);
                            };

                            return (
                                <div className={styles.configPanel}>
                                    <div className={styles.configHeader}>
                                        <h2>{profile.name}</h2>
                                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                                            <button
                                                className={styles.saveBtn}
                                                style={{ background: 'var(--color-accent-primary)', padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                                                onClick={() => handleScan(profile.id)}
                                                disabled={isScanning || !profile.sourcePath || profile.streams.length === 0}
                                            >
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                                    <BsSearch /> {isScanning ? 'Scanning...' : 'Scan & Validate'}
                                                </span>
                                            </button>
                                            <button
                                                className={styles.saveBtn}
                                                style={{ background: 'var(--color-danger)', padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                                                onClick={() => {
                                                    removeProfile(profile.id);
                                                    setSelectedProfileId(null);
                                                }}
                                            >Delete</button>
                                        </div>
                                    </div>

                                    {/* Source Path */}
                                    <div className={styles.configSection}>
                                        <h3>Source</h3>
                                        <div className={styles.configRow}>
                                            <div className={styles.inputGroup} style={{ flex: 1 }}>
                                                <label>Root Directory</label>
                                                <input type="text" readOnly value={profile.sourcePath || 'Not set'} />
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Total Samples</label>
                                                <input type="text" readOnly value={profile.totalSamples !== undefined ? profile.totalSamples : "Unscanned"} style={{ width: '100px' }} />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Scan Result Status */}
                                    {profile.scanResult && (
                                        <div className={styles.configSection} style={{
                                            borderColor: profile.scanResult.droppedCount > 0 ? 'var(--color-warning)' : 'var(--color-success)',
                                            borderWidth: '2px'
                                        }}>
                                            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                {profile.scanResult.droppedCount === 0
                                                    ? <><BsCheckCircle color="var(--color-success)" /> Scan Result — All Streams Aligned</>
                                                    : <><BsExclamationTriangle color="var(--color-warning)" /> Scan Result — {profile.scanResult.droppedCount} Samples Dropped</>
                                                }
                                            </h3>
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
                                                <div style={{ display: 'flex', gap: '2rem' }}>
                                                    <span><strong>Matched:</strong> {profile.scanResult.totalMatched}</span>
                                                    <span><strong>Dropped:</strong> {profile.scanResult.droppedCount}</span>
                                                    <span style={{ color: 'var(--color-text-secondary)' }}>
                                                        Scanned at {new Date(profile.scanResult.timestamp).toLocaleTimeString()}
                                                    </span>
                                                </div>
                                                {profile.scanResult.streamReports.map(report => (
                                                    <div key={report.streamId} style={{
                                                        display: 'flex', gap: '1rem', alignItems: 'center',
                                                        padding: '0.4rem 0.6rem',
                                                        background: 'var(--color-bg-primary)',
                                                        borderRadius: '4px'
                                                    }}>
                                                        <span style={{ fontWeight: 500 }}>{report.alias}</span>
                                                        <span style={{ color: 'var(--color-success)' }}>✓ {report.foundCount} found</span>
                                                        {report.missingSampleIds.length > 0 && (
                                                            <span style={{ color: 'var(--color-warning)' }}>
                                                                ✗ {report.missingSampleIds.length} missing
                                                            </span>
                                                        )}
                                                        {report.discoveredClasses && (
                                                            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
                                                                {Object.entries(report.discoveredClasses)
                                                                    .sort((a, b) => b[1] - a[1])
                                                                    .map(([cls, count]) => (
                                                                        <span key={cls} style={{
                                                                            background: 'var(--color-bg-tertiary)',
                                                                            padding: '0.15rem 0.5rem',
                                                                            borderRadius: '12px',
                                                                            fontSize: '0.8rem',
                                                                            border: '1px solid var(--color-border)',
                                                                        }}>
                                                                            {cls}: {count}
                                                                        </span>
                                                                    ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Data Streams */}
                                    <DataStreamsPanel profile={profile} />

                                    {/* Splits */}
                                    <div className={styles.configSection}>
                                        <h3>Dataset Splits</h3>
                                        <div className={styles.configRow}>
                                            <div className={styles.inputGroup}>
                                                <label>Train (%)</label>
                                                <input
                                                    type="number" min="0" max="100"
                                                    value={profile.split.train}
                                                    onChange={e => handleUpdate({ split: { ...profile.split, train: Number(e.target.value) } })}
                                                />
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Validation (%)</label>
                                                <input
                                                    type="number" min="0" max="100"
                                                    value={profile.split.val}
                                                    onChange={e => handleUpdate({ split: { ...profile.split, val: Number(e.target.value) } })}
                                                />
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Test (%)</label>
                                                <input
                                                    type="number" min="0" max="100"
                                                    value={profile.split.test}
                                                    onChange={e => handleUpdate({ split: { ...profile.split, test: Number(e.target.value) } })}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                </div>
                            );
                        })() : (
                            <div className={styles.placeholder}>
                                <BsDatabaseAdd className={styles.placeholderIcon} />
                                <h3>No Dataset Selected</h3>
                                <p>Select an existing profile from the list or create a new one.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {isCreateModalOpen && <CreateDatasetModal onClose={() => setIsCreateModalOpen(false)} />}
        </>
    );
};
