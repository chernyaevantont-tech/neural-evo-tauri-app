import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './GenomeLibraryPage.module.css';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import { BsCollection, BsBoxArrowUpRight, BsTrash, BsSearch, BsFileEarmarkPlus } from 'react-icons/bs';
import { useGenomeLibraryStore, GenomeLibraryEntry, CompatibilityStatus } from '../../features/genome-library';
import { checkCompatibility } from '../../features/genome-library/lib/compatibility';
import { GenomePreviewCanvas } from '../../features/genome-library/ui/GenomePreviewCanvas';
import { useDatasetManagerStore } from '../../features/dataset-manager/model/store';
import { loadGenomeApi } from '../../features/genome-save-load/api/loadGenome';
import { useCanvasGenomeStore } from '../../entities/canvas-genome/model/store';
import { useCanvasStateStore } from '../../entities/canvas-state/model/store';
import { deserializeGenome } from '../../entities/canvas-genome/lib/deserializeGenome';

type SortKey = 'date' | 'name' | 'nodes';

export const GenomeLibraryPage: React.FC = () => {
    const navigate = useNavigate();
    const { entries, isLoading, loadLibrary, deleteGenome, saveGenome, loadGenomeContent } = useGenomeLibraryStore();
    const resetCanvasState = useCanvasStateStore(s => s.reset);
    const resetCanvasGenome = useCanvasGenomeStore(s => s.reset);
    const addGenome = useCanvasGenomeStore(s => s.addGenome);
    const datasetProfiles = useDatasetManagerStore(s => s.profiles);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [search, setSearch] = useState('');
    const [sortKey, setSortKey] = useState<SortKey>('date');
    const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);

    useEffect(() => {
        loadLibrary();
    }, [loadLibrary]);

    // Auto-select first dataset profile if available
    useEffect(() => {
        if (!selectedDatasetId && datasetProfiles.length > 0) {
            setSelectedDatasetId(datasetProfiles[0].id);
        }
    }, [datasetProfiles, selectedDatasetId]);

    const selectedDataset = datasetProfiles.find(p => p.id === selectedDatasetId);

    // Filter & Sort
    const filteredEntries = useMemo(() => {
        let result = entries.filter(e => {
            const q = search.toLowerCase();
            return e.name.toLowerCase().includes(q) ||
                e.tags.some(t => t.toLowerCase().includes(q)) ||
                e.layerTypes.some(l => l.toLowerCase().includes(q));
        });

        result.sort((a, b) => {
            switch (sortKey) {
                case 'date': return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
                case 'name': return a.name.localeCompare(b.name);
                case 'nodes': return a.totalNodes - b.totalNodes;
                default: return 0;
            }
        });
        return result;
    }, [entries, search, sortKey]);

    const selectedEntry = entries.find(e => e.id === selectedId);

    const getCompat = (entry: GenomeLibraryEntry): CompatibilityStatus | null => {
        if (!selectedDataset) return null;
        return checkCompatibility(entry, selectedDataset.streams);
    };

    const compatLabel = (status: CompatibilityStatus | null) => {
        if (!status) return null;
        const labels: Record<CompatibilityStatus, string> = {
            compatible: '✅ Compatible',
            adaptable: '⚠ Adaptable',
            incompatible: '❌ Incompatible',
        };
        return (
            <span className={`${styles.compatBadge} ${styles[status]}`}>
                {labels[status]}
            </span>
        );
    };

    const dimsStr = (dims: number[]) => dims.map(d => `${d}D`).join(', ');

    const handleDelete = async (id: string) => {
        await deleteGenome(id);
        if (selectedId === id) setSelectedId(null);
    };

    const handleOpenInSandbox = async (id: string) => {
        try {
            const genomeStr = await loadGenomeContent(id);
            const { nodes, genome } = await deserializeGenome(genomeStr);
            resetCanvasState();
            resetCanvasGenome();
            addGenome(nodes, genome, 800, 600, 0, 0, 1, 300);
            navigate('/sandbox');
        } catch (err) {
            console.error('Failed to open in sandbox:', err);
        }
    };

    const handleImport = async () => {
        try {
            const genomeStr = await loadGenomeApi();
            const name = prompt('Genome name:', 'Imported Genome') || 'Imported Genome';
            const tagsStr = prompt('Tags (comma separated):', '') || '';
            const tags = tagsStr.split(',').map(t => t.trim()).filter(Boolean);
            await saveGenome(genomeStr, name, tags);
        } catch (err) {
            console.error('Import failed:', err);
        }
    };

    return (
        <>
            <TitleBar />
            <div className={styles.container}>
                <div className={styles.content}>
                    {/* Left Sidebar: Genome List */}
                    <div className={styles.sidebar}>
                        <div className={styles.sidebarHeader}>
                            <h2 className={styles.sidebarTitle}>🧬 Genome Library</h2>
                            <button className={styles.addButton} onClick={handleImport} title="Import from file">
                                <BsFileEarmarkPlus />
                            </button>
                        </div>

                        {/* Search & Sort */}
                        <div className={styles.searchRow}>
                            <input
                                className={styles.searchInput}
                                placeholder="Search name, tag, layer..."
                                value={search}
                                onChange={e => setSearch(e.target.value)}
                            />
                            <select
                                className={styles.sortSelect}
                                value={sortKey}
                                onChange={e => setSortKey(e.target.value as SortKey)}
                            >
                                <option value="date">Date ↓</option>
                                <option value="name">Name</option>
                                <option value="nodes">Nodes</option>
                            </select>
                        </div>

                        {/* Dataset context for compatibility */}
                        {datasetProfiles.length > 0 && (
                            <select
                                className={styles.sortSelect}
                                value={selectedDatasetId || ''}
                                onChange={e => setSelectedDatasetId(e.target.value || null)}
                                style={{ width: '100%' }}
                            >
                                <option value="">No dataset (skip compat)</option>
                                {datasetProfiles.map(p => (
                                    <option key={p.id} value={p.id}>{p.name}</option>
                                ))}
                            </select>
                        )}

                        {/* Genome List */}
                        <div className={styles.genomeList}>
                            {isLoading && <div style={{ textAlign: 'center', opacity: 0.5 }}>Loading...</div>}
                            {!isLoading && filteredEntries.length === 0 && (
                                <div className={styles.emptyLibrary}>
                                    <BsCollection style={{ fontSize: '2rem' }} />
                                    <span>Library is empty</span>
                                    <span style={{ fontSize: '0.8rem' }}>Save genomes from Sandbox to see them here</span>
                                </div>
                            )}
                            {filteredEntries.map(entry => {
                                const compat = getCompat(entry);
                                return (
                                    <div
                                        key={entry.id}
                                        className={`${styles.genomeItem} ${selectedId === entry.id ? styles.active : ''}`}
                                        onClick={() => setSelectedId(entry.id)}
                                    >
                                        <div className={styles.genomeName}>{entry.name}</div>
                                        <div className={styles.genomeMeta}>
                                            <span>{dimsStr(entry.inputDims)}→{dimsStr(entry.outputDims)}</span>
                                            <span>·</span>
                                            <span>{entry.totalNodes} nodes</span>
                                            {compat && <>{compatLabel(compat)}</>}
                                        </div>
                                        {entry.tags.length > 0 && (
                                            <div className={styles.genomeTagsList}>
                                                {entry.tags.map(t => (
                                                    <span key={t} className={styles.tag}>{t}</span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Right Panel: Detail & Preview */}
                    <div className={styles.mainPanel}>
                        {selectedEntry ? (
                            <>
                                {/* Header */}
                                <div className={styles.detailHeader}>
                                    <h2 className={styles.detailTitle}>{selectedEntry.name}</h2>
                                    <div className={styles.detailActions}>
                                        <button
                                            className={`${styles.actionBtn} ${styles.primaryBtn}`}
                                            onClick={() => handleOpenInSandbox(selectedEntry.id)}
                                        >
                                            <BsBoxArrowUpRight /> Open in Sandbox
                                        </button>
                                        <button
                                            className={`${styles.actionBtn} ${styles.dangerBtn}`}
                                            onClick={() => handleDelete(selectedEntry.id)}
                                        >
                                            <BsTrash /> Delete
                                        </button>
                                    </div>
                                </div>

                                {/* Info Cards */}
                                <div className={styles.infoGrid}>
                                    <div className={styles.infoCard}>
                                        <div className={styles.infoLabel}>Input Dims</div>
                                        <div className={styles.infoValue}>{dimsStr(selectedEntry.inputDims)}</div>
                                    </div>
                                    <div className={styles.infoCard}>
                                        <div className={styles.infoLabel}>Output Dims</div>
                                        <div className={styles.infoValue}>{dimsStr(selectedEntry.outputDims)}</div>
                                    </div>
                                    <div className={styles.infoCard}>
                                        <div className={styles.infoLabel}>Total Nodes</div>
                                        <div className={styles.infoValue}>{selectedEntry.totalNodes}</div>
                                    </div>
                                    <div className={styles.infoCard}>
                                        <div className={styles.infoLabel}>Created</div>
                                        <div className={styles.infoValue}>
                                            {new Date(selectedEntry.createdAt).toLocaleDateString()}
                                        </div>
                                    </div>
                                    {selectedEntry.bestLoss !== undefined && (
                                        <div className={styles.infoCard}>
                                            <div className={styles.infoLabel}>Best Loss</div>
                                            <div className={styles.infoValue}>{selectedEntry.bestLoss?.toFixed(4)}</div>
                                        </div>
                                    )}
                                    {selectedEntry.bestAccuracy !== undefined && (
                                        <div className={styles.infoCard}>
                                            <div className={styles.infoLabel}>Best Accuracy</div>
                                            <div className={styles.infoValue}>{(selectedEntry.bestAccuracy! * 100).toFixed(1)}%</div>
                                        </div>
                                    )}
                                </div>

                                {/* Compatibility */}
                                {selectedDataset && (
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                        <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                                            Compatibility with <strong>{selectedDataset.name}</strong>:
                                        </span>
                                        {compatLabel(getCompat(selectedEntry))}
                                    </div>
                                )}

                                {/* Layer Types */}
                                <div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem' }}>
                                        LAYER TYPES
                                    </div>
                                    <div className={styles.layerTypesList}>
                                        {selectedEntry.layerTypes.map(lt => (
                                            <span key={lt} className={styles.layerBadge}>{lt}</span>
                                        ))}
                                    </div>
                                </div>

                                {/* Tags */}
                                {selectedEntry.tags.length > 0 && (
                                    <div>
                                        <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem' }}>
                                            TAGS
                                        </div>
                                        <div className={styles.genomeTagsList}>
                                            {selectedEntry.tags.map(t => (
                                                <span key={t} className={styles.tag}>{t}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Preview Canvas */}
                                <div className={styles.previewContainer}>
                                    <GenomePreviewCanvas genomeId={selectedEntry.id} />
                                </div>
                            </>
                        ) : (
                            <div className={styles.placeholder} style={{ flex: 1 }}>
                                <BsCollection className={styles.placeholderIcon} />
                                <h3>Select a Genome</h3>
                                <p>Choose a genome from the list to view details and preview.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
};
