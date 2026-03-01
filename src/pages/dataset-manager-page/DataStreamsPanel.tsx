import React from 'react';
import styles from './DatasetManagerPage.module.css';
import { DataStream, DatasetProfile, useDatasetManagerStore, VisionSettings, TabularSettings, defaultVisionSettings } from '../../features/dataset-manager/model/store';
import { BsPlusLg, BsTrash, BsBoxArrowInRight, BsBullseye, BsGearFill } from 'react-icons/bs';

interface Props {
    profile: DatasetProfile;
}

const defaultTabularSettings: TabularSettings = {
    normalization: 'min-max',
    oneHot: false,
    fillMissing: 'mean'
};

export const DataStreamsPanel: React.FC<Props> = ({ profile }) => {
    const updateProfile = useDatasetManagerStore(s => s.updateProfile);

    const updateStream = (streamId: string, patch: Partial<DataStream>) => {
        const newStreams = profile.streams.map(s => s.id === streamId ? { ...s, ...patch } : s);
        updateProfile(profile.id, { streams: newStreams });
    };

    const handleAddStream = () => {
        const newStream: DataStream = {
            id: crypto.randomUUID(),
            alias: `New Stream ${profile.streams.length + 1}`,
            role: 'Input',
            dataType: 'Image',
            tensorShape: [],
            locator: { type: 'GlobPattern', pattern: '**/*.jpg' },
            preprocessing: { vision: { ...defaultVisionSettings } }
        };
        updateProfile(profile.id, { streams: [...profile.streams, newStream] });
    };

    const handleRemoveStream = (id: string) => {
        updateProfile(profile.id, { streams: profile.streams.filter(s => s.id !== id) });
    };

    return (
        <div className={styles.configSection}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0 }}>Data Streams</h3>
                <button
                    onClick={handleAddStream}
                    style={{
                        background: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border)',
                        color: 'var(--color-text-primary)',
                        padding: '0.4rem 0.8rem',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        display: 'flex', gap: '0.5rem', alignItems: 'center'
                    }}>
                    <BsPlusLg /> Add Stream
                </button>
            </div>

            {profile.streams.length === 0 ? (
                <div className={styles.placeholder} style={{ padding: '2rem 0' }}>
                    <p style={{ margin: 0 }}>No data streams defined yet.</p>
                    <small>Add an Input stream (e.g. Images) and Target streams (e.g. Classes) to configure your dataset.</small>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {profile.streams.map(stream => (
                        <div key={stream.id} style={{
                            background: 'var(--color-bg-primary)',
                            border: '1px solid var(--color-border)',
                            borderRadius: '8px',
                            padding: '1rem',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '1rem'
                        }}>
                            {/* Stream Header */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                    {stream.role === 'Input' ? <BsBoxArrowInRight color="var(--color-accent-primary)" /> : <BsBullseye color="var(--color-danger)" />}
                                    <input
                                        type="text"
                                        value={stream.alias}
                                        onChange={(e) => updateStream(stream.id, { alias: e.target.value })}
                                        style={{
                                            background: 'transparent', border: 'none', borderBottom: '1px solid var(--color-border)', color: 'white', fontSize: '1.1rem', fontWeight: 600, outline: 'none'
                                        }}
                                    />
                                    <select
                                        value={stream.role}
                                        onChange={(e) => updateStream(stream.id, { role: e.target.value as any })}
                                        style={{ background: 'var(--color-bg-tertiary)', color: 'white', border: 'none', padding: '0.2rem 0.5rem', borderRadius: '4px' }}
                                    >
                                        <option value="Input">Input Layer</option>
                                        <option value="Target">Target (Loss)</option>
                                        <option value="Ignore">Ignore</option>
                                    </select>
                                </div>
                                <BsTrash
                                    color="var(--color-text-muted)"
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleRemoveStream(stream.id)}
                                />
                            </div>

                            {/* Stream Config row */}
                            <div className={styles.configRow}>
                                <div className={styles.inputGroup} style={{ flex: 1 }}>
                                    <label>Data Type</label>
                                    <select
                                        value={stream.dataType}
                                        onChange={(e) => {
                                            const dataType = e.target.value as any;
                                            // Auto-populate default preprocessing for the chosen dataType
                                            let preprocessing = stream.preprocessing;
                                            if (dataType === 'Image') {
                                                preprocessing = { vision: { ...defaultVisionSettings } };
                                            } else if (dataType === 'Vector') {
                                                preprocessing = { tabular: { ...defaultTabularSettings } };
                                            } else {
                                                preprocessing = undefined;
                                            }
                                            updateStream(stream.id, { dataType, preprocessing });
                                        }}
                                    >
                                        <option value="Image">Image (Tensor 4D)</option>
                                        <option value="Vector">Vector (Tensor 2D)</option>
                                        <option value="Categorical">Categorical (Class Label)</option>
                                        <option value="Text">Text (Tokens)</option>
                                    </select>
                                </div>
                                <div className={styles.inputGroup} style={{ flex: 2 }}>
                                    <label>Location Rule (Locator)</label>
                                    <select
                                        value={stream.locator.type}
                                        onChange={(e) => {
                                            const type = e.target.value as any;
                                            let newLocator: any = { type: 'None' };
                                            if (type === 'GlobPattern') newLocator = { type: 'GlobPattern', pattern: '**/*.jpg' };
                                            if (type === 'FolderMapping') newLocator = { type: 'FolderMapping' };
                                            if (type === 'CompanionFile') newLocator = { type: 'CompanionFile', pathTemplate: '../labels/{id}.txt', parser: 'YOLO' };

                                            updateStream(stream.id, { locator: newLocator });
                                        }}
                                    >
                                        <option value="GlobPattern">Glob Pattern (Search Files)</option>
                                        <option value="FolderMapping">Folder Mapping (Class from Parent Dir)</option>
                                        <option value="CompanionFile">Companion File (Neighbor file)</option>
                                        <option value="None">None / Manual</option>
                                    </select>
                                </div>
                            </div>

                            {/* Locator Args */}
                            {stream.locator.type === 'GlobPattern' && (
                                <div className={styles.inputGroup}>
                                    <label>File Pattern Mask</label>
                                    <input
                                        type="text"
                                        value={stream.locator.pattern}
                                        onChange={(e) => updateStream(stream.id, { locator: { ...stream.locator, pattern: e.target.value } as any })}
                                        placeholder="e.g. images/**/*.jpg"
                                    />
                                </div>
                            )}
                            {stream.locator.type === 'CompanionFile' && (
                                <div className={styles.configRow}>
                                    <div className={styles.inputGroup} style={{ flex: 2 }}>
                                        <label>Path Template</label>
                                        <input
                                            type="text"
                                            value={stream.locator.pathTemplate}
                                            onChange={(e) => updateStream(stream.id, { locator: { ...stream.locator, pathTemplate: e.target.value } as any })}
                                            placeholder="e.g. ../labels/{id}.txt"
                                        />
                                    </div>
                                    <div className={styles.inputGroup} style={{ flex: 1 }}>
                                        <label>Parser</label>
                                        <select
                                            value={stream.locator.parser}
                                            onChange={(e) => updateStream(stream.id, { locator: { ...stream.locator, parser: e.target.value } as any })}
                                        >
                                            <option value="YOLO">YOLO (class x y w h)</option>
                                            <option value="Text">Plain Text</option>
                                            <option value="COCO_Subset">COCO JSON Subset</option>
                                        </select>
                                    </div>
                                </div>
                            )}

                            {/* Per-stream Preprocessing: Vision */}
                            {stream.dataType === 'Image' && stream.preprocessing?.vision && (() => {
                                const v = stream.preprocessing.vision!;
                                const updateVision = (patch: Partial<VisionSettings>) => {
                                    updateStream(stream.id, { preprocessing: { ...stream.preprocessing, vision: { ...v, ...patch } } });
                                };
                                return (
                                    <div style={{ borderTop: '1px dashed var(--color-border)', paddingTop: '0.75rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                                            <BsGearFill /> Preprocessing & Augmentation
                                        </div>
                                        <div className={styles.configRow}>
                                            <div className={styles.inputGroup}>
                                                <label>Resize W</label>
                                                <input type="number" min="1" value={v.resize[0]} onChange={e => updateVision({ resize: [Number(e.target.value), v.resize[1]] })} style={{ width: '70px' }} />
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Resize H</label>
                                                <input type="number" min="1" value={v.resize[1]} onChange={e => updateVision({ resize: [v.resize[0], Number(e.target.value)] })} style={{ width: '70px' }} />
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Normalization</label>
                                                <select value={v.normalization} onChange={e => updateVision({ normalization: e.target.value as any })}>
                                                    <option value="0-1">0..1 (MinMax)</option>
                                                    <option value="imagenet">ImageNet</option>
                                                    <option value="none">None</option>
                                                </select>
                                            </div>
                                            <div className={`${styles.inputGroup} ${styles.checkboxRow}`}>
                                                <input type="checkbox" checked={v.grayscale} onChange={e => updateVision({ grayscale: e.target.checked })} />
                                                <label>Grayscale</label>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })()}

                            {/* Per-stream Preprocessing: Tabular */}
                            {stream.dataType === 'Vector' && stream.preprocessing?.tabular && (() => {
                                const t = stream.preprocessing.tabular!;
                                const updateTabular = (patch: Partial<TabularSettings>) => {
                                    updateStream(stream.id, { preprocessing: { ...stream.preprocessing, tabular: { ...t, ...patch } } });
                                };
                                return (
                                    <div style={{ borderTop: '1px dashed var(--color-border)', paddingTop: '0.75rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                                            <BsGearFill /> Preprocessing
                                        </div>
                                        <div className={styles.configRow}>
                                            <div className={styles.inputGroup}>
                                                <label>Normalization</label>
                                                <select value={t.normalization} onChange={e => updateTabular({ normalization: e.target.value as any })}>
                                                    <option value="min-max">Min-Max</option>
                                                    <option value="z-score">Z-Score</option>
                                                    <option value="none">None</option>
                                                </select>
                                            </div>
                                            <div className={styles.inputGroup}>
                                                <label>Fill Missing</label>
                                                <select value={t.fillMissing} onChange={e => updateTabular({ fillMissing: e.target.value as any })}>
                                                    <option value="mean">Mean</option>
                                                    <option value="median">Median</option>
                                                    <option value="mode">Mode</option>
                                                    <option value="drop">Drop Row</option>
                                                </select>
                                            </div>
                                            <div className={`${styles.inputGroup} ${styles.checkboxRow}`}>
                                                <input type="checkbox" checked={t.oneHot} onChange={e => updateTabular({ oneHot: e.target.checked })} />
                                                <label>One-Hot Encode</label>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })()}

                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
