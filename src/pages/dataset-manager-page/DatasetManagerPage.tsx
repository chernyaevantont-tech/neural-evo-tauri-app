import React from 'react';
import styles from './DatasetManagerPage.module.css';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import { BsDatabaseAdd, BsFolder2Open, BsFiletypeCsv, BsSearch, BsCheckCircle, BsExclamationTriangle, BsLightningCharge } from 'react-icons/bs';
import { useDatasetManagerStore, DatasetSourceType, ScanResult } from '../../features/dataset-manager/model/store';
import { CreateDatasetModal } from './CreateDatasetModal';
import { DataStreamsPanel } from './DataStreamsPanel';
import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

export const DatasetManagerPage: React.FC = () => {
    const { profiles, selectedProfileId, setSelectedProfileId, removeProfile } = useDatasetManagerStore();
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isScanning, setIsScanning] = useState(false);
    const [isCaching, setIsCaching] = useState(false);

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
            const streamConfigs = profile.streams.map(s => {
                const config: any = {
                    stream_id: s.id,
                    alias: s.alias,
                    locator_type: s.locator.type,
                    pattern: s.locator.type === 'GlobPattern' ? s.locator.pattern : null,
                    path_template: s.locator.type === 'CompanionFile' ? s.locator.pathTemplate : null,
                    stream_role: s.role, // ← Send the role
                };
                
                // Add CSV-specific parameters if this is a CSV Dataset locator
                if (s.locator.type === 'CsvDataset') {
                    config.csv_path = s.locator.csvPath;
                    config.has_headers = s.locator.hasHeaders;
                    config.sample_mode = s.locator.sampleMode;
                    // Only send feature_columns for Input streams, empty for Target streams
                    config.feature_columns = s.role === 'Input' ? s.locator.featureColumns : [];
                    // Only send target_column for Target streams, empty for Input streams
                    config.target_column = s.role === 'Target' ? s.locator.targetColumn : '';
                    config.window_size = s.locator.windowSize ?? null;
                }
                
                return config;
            });

            const result = await invoke<{
                total_matched: number;
                dropped_count: number;
                stream_reports: Array<{
                    stream_id: string;
                    alias: string;
                    found_count: number;
                    missing_sample_ids: string[];
                    discovered_classes: Record<string, number> | null;
                    input_shape: number[] | null;
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
                    inputShape: r.input_shape ?? undefined,
                })),
                timestamp: new Date().toISOString(),
            };

            // Update profile streams with tensorShape and num_classes from scan results
            const updatedProfile = { ...profile };
            updatedProfile.streams = profile.streams.map(stream => {
                const report = result.stream_reports.find(r => r.stream_id === stream.id);
                if (report) {
                    return {
                        ...stream,
                        // Set tensorShape from input_shape for Input streams
                        tensorShape: report.input_shape ? report.input_shape : stream.tensorShape,
                        // Set num_classes for Target streams from discovered_classes
                        numClasses: report.discovered_classes ? Object.keys(report.discovered_classes).length : stream.numClasses,
                    } as any;
                }
                return stream;
            });

            useDatasetManagerStore.getState().updateProfile(profileId, {
                scanResult,
                totalSamples: scanResult.totalMatched,
                isScanned: true,
                streams: updatedProfile.streams,
            });
        } catch (err) {
            console.error('Scan failed:', err);
        } finally {
            setIsScanning(false);
        }
    };

    const handleCache = async (profileId: string) => {
        setIsCaching(true);
        try {
            const result = await invoke<{
                total_cached: number;
                total_dropped: number;
                dropped_sample_ids: string[];
                class_counts: Record<string, number>;
            }>('cache_dataset', { datasetProfileId: profileId });

            const profile = useDatasetManagerStore.getState().profiles.find(p => p.id === profileId);
            if (profile) {
                // Build updated stream reports with real per-class counts from cache
                const updatedStreamReports = (profile.scanResult?.streamReports || []).map(report => {
                    if (report.discoveredClasses) {
                        // Replace with actual per-class counts from cached data
                        return {
                            ...report,
                            foundCount: result.total_cached,
                            discoveredClasses: result.class_counts,
                        };
                    }
                    return report;
                });

                const updatedScanResult: ScanResult = {
                    totalMatched: result.total_cached,
                    droppedCount: result.total_dropped,
                    streamReports: updatedStreamReports,
                    timestamp: new Date().toISOString(),
                };

                useDatasetManagerStore.getState().updateProfile(profileId, {
                    scanResult: updatedScanResult,
                    totalSamples: result.total_cached,
                });
            }
        } catch (err) {
            console.error('Caching failed:', err);
        } finally {
            setIsCaching(false);
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
                                                disabled={isScanning || isCaching || !profile.sourcePath || profile.streams.length === 0}
                                            >
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                                    <BsSearch /> {isScanning ? 'Scanning...' : 'Scan & Validate'}
                                                </span>
                                            </button>
                                            <button
                                                className={styles.saveBtn}
                                                style={{ background: 'var(--color-accent-secondary)', padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                                                onClick={() => handleCache(profile.id)}
                                                disabled={isScanning || isCaching || !profile.isScanned}
                                            >
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                                    <BsLightningCharge /> {isCaching ? 'Caching...' : 'Build AoT Cache'}
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
                                                    <div style={{ marginTop: '0.5rem' }}>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                                                            <strong style={{ fontSize: '0.85rem' }}>Class Distribution ({report.alias})</strong>
                                                            <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                                                                {Object.keys(report.discoveredClasses).length} classes
                                                            </span>
                                                        </div>
                                                        <div style={{
                                                            display: 'flex', flexDirection: 'column', gap: '4px',
                                                            background: 'var(--color-bg-tertiary)', padding: '8px', borderRadius: '4px'
                                                        }}>
                                                            {(() => {
                                                                const entries = Object.entries(report.discoveredClasses).sort((a, b) => b[1] - a[1]);
                                                                const maxCount = Math.max(...entries.map(e => e[1]));
                                                                return entries.map(([cls, count]) => (
                                                                    <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                                        <div style={{ width: '100px', fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={cls}>{cls}</div>
                                                                        <div style={{ flex: 1, height: '8px', background: 'var(--color-bg-primary)', borderRadius: '4px', overflow: 'hidden' }}>
                                                                            <div style={{
                                                                                width: `${(count / maxCount) * 100}%`,
                                                                                height: '100%',
                                                                                background: count / maxCount < 0.2 ? 'var(--color-warning)' : 'var(--color-accent-primary)',
                                                                                transition: 'width 0.3s ease'
                                                                            }} />
                                                                        </div>
                                                                        <div style={{ width: '40px', fontSize: '0.75rem', textAlign: 'right' }}>{count}</div>
                                                                    </div>
                                                                ));
                                                            })()}
                                                        </div>
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
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                            <h3 style={{ margin: 0 }}>Dataset Splits</h3>
                                            {profile.streams.some(s => s.role === 'Target' && s.dataType === 'Categorical') && (
                                                <span style={{
                                                    fontSize: '0.75rem',
                                                    color: 'var(--color-success)',
                                                    background: 'rgba(52, 211, 153, 0.1)',
                                                    padding: '2px 8px',
                                                    borderRadius: '10px',
                                                    border: '1px solid var(--color-success)'
                                                }}>
                                                    Stratified (Classification)
                                                </span>
                                            )}
                                        </div>
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
