import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BsArchive, BsArrowClockwise, BsTrash, BsBoxArrowUpRight, BsSearch, BsInfoCircle } from 'react-icons/bs';
import styles from './HiddenArchivePage.module.css';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import {
    useGenomeLibraryStore,
    type GenomeLibraryEntry,
} from '../../features/genome-library';
import { ExportGenomeWithWeightsModal } from '../../features/evolution-studio/ui/ExportGenomeWithWeightsModal';

type SortKey = 'created_desc' | 'generation_desc' | 'accuracy_desc' | 'latency_asc' | 'model_size_asc';

function toNumber(value: string): number | undefined {
    if (!value.trim()) {
        return undefined;
    }
    const n = Number(value);
    return Number.isFinite(n) ? n : undefined;
}

function metricAccuracy(entry: GenomeLibraryEntry): number {
    return entry.fitnessMetrics?.accuracy ?? entry.bestAccuracy ?? 0;
}

function metricLatency(entry: GenomeLibraryEntry): number {
    return entry.fitnessMetrics?.inferenceLatencyMs ?? Number.POSITIVE_INFINITY;
}

function metricModelSize(entry: GenomeLibraryEntry): number {
    return entry.fitnessMetrics?.modelSizeMb ?? Number.POSITIVE_INFINITY;
}

export function filterAndSortHiddenEntries(entries: GenomeLibraryEntry[], searchId: string, sortKey: SortKey): GenomeLibraryEntry[] {
    const query = searchId.trim().toLowerCase();

    const filtered = entries.filter((entry) => {
        if (!query) {
            return true;
        }
        return entry.id.toLowerCase().includes(query) || entry.name.toLowerCase().includes(query);
    });

    filtered.sort((a, b) => {
        switch (sortKey) {
            case 'generation_desc':
                return (b.sourceGeneration ?? 0) - (a.sourceGeneration ?? 0);
            case 'accuracy_desc':
                return metricAccuracy(b) - metricAccuracy(a);
            case 'latency_asc':
                return metricLatency(a) - metricLatency(b);
            case 'model_size_asc':
                return metricModelSize(a) - metricModelSize(b);
            case 'created_desc':
            default:
                return (b.createdAtUnixMs ?? 0) - (a.createdAtUnixMs ?? 0);
        }
    });

    return filtered;
}

export const HiddenArchivePage: React.FC = () => {
    const navigate = useNavigate();
    const {
        listHiddenLibrary,
        unhideHiddenGenome,
        deleteHiddenGenome,
        exportGenomeWithWeights,
        pickFolder,
        getGenealogyPath,
    } = useGenomeLibraryStore();

    const [entries, setEntries] = useState<GenomeLibraryEntry[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [operationResult, setOperationResult] = useState<string | null>(null);

    const [searchId, setSearchId] = useState('');
    const [sortKey, setSortKey] = useState<SortKey>('created_desc');

    const [generationMin, setGenerationMin] = useState('');
    const [generationMax, setGenerationMax] = useState('');
    const [accuracyMin, setAccuracyMin] = useState('');
    const [accuracyMax, setAccuracyMax] = useState('');
    const [latencyMin, setLatencyMin] = useState('');
    const [latencyMax, setLatencyMax] = useState('');

    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [activeDetailId, setActiveDetailId] = useState<string | null>(null);
    const [lineageRecords, setLineageRecords] = useState<string[]>([]);
    const [isLineageLoading, setIsLineageLoading] = useState(false);
    const [exportGenomeId, setExportGenomeId] = useState<string | null>(null);

    const loadEntries = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const result = await listHiddenLibrary({
                generationMin: toNumber(generationMin),
                generationMax: toNumber(generationMax),
                accuracyMin: toNumber(accuracyMin),
                accuracyMax: toNumber(accuracyMax),
                latencyMinMs: toNumber(latencyMin),
                latencyMaxMs: toNumber(latencyMax),
            });
            setEntries(result);
            setSelectedIds((prev) => {
                const next = new Set<string>();
                result.forEach((entry) => {
                    if (prev.has(entry.id)) {
                        next.add(entry.id);
                    }
                });
                return next;
            });
        } catch (e) {
            setError(String(e));
        } finally {
            setIsLoading(false);
        }
    }, [accuracyMax, accuracyMin, generationMax, generationMin, latencyMax, latencyMin, listHiddenLibrary]);

    useEffect(() => {
        loadEntries();
    }, [loadEntries]);

    const visibleEntries = useMemo(
        () => filterAndSortHiddenEntries(entries, searchId, sortKey),
        [entries, searchId, sortKey],
    );

    const allVisibleSelected = visibleEntries.length > 0 && visibleEntries.every((entry) => selectedIds.has(entry.id));

    const selectedCount = selectedIds.size;

    const activeDetail = useMemo(
        () => entries.find((entry) => entry.id === activeDetailId) ?? null,
        [activeDetailId, entries],
    );

    const toggleSelectOne = (id: string) => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    const toggleSelectAllVisible = () => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (allVisibleSelected) {
                visibleEntries.forEach((entry) => next.delete(entry.id));
            } else {
                visibleEntries.forEach((entry) => next.add(entry.id));
            }
            return next;
        });
    };

    const runBatch = async (
        label: string,
        executor: (id: string) => Promise<void>,
        confirmation: string,
    ) => {
        if (selectedCount === 0) {
            setOperationResult(`No rows selected for ${label.toLowerCase()}.`);
            return;
        }

        if (!window.confirm(confirmation)) {
            return;
        }

        const ids = Array.from(selectedIds);
        let success = 0;
        let failed = 0;

        for (const id of ids) {
            try {
                await executor(id);
                success += 1;
            } catch {
                failed += 1;
            }
        }

        setOperationResult(`${label}: success ${success}, failed ${failed}`);
        setSelectedIds(new Set());
        await loadEntries();
    };

    const handleBatchUnhide = () => runBatch(
        'Unhide selected',
        unhideHiddenGenome,
        `Unhide ${selectedCount} selected genomes?`,
    );

    const handleBatchDelete = () => runBatch(
        'Delete selected',
        deleteHiddenGenome,
        `Delete ${selectedCount} selected genomes permanently?`,
    );

    const handleBatchExport = async () => {
        if (selectedCount === 0) {
            setOperationResult('No rows selected for export.');
            return;
        }

        if (!window.confirm(`Export ${selectedCount} selected genomes?`)) {
            return;
        }

        const outputPath = await pickFolder();
        if (!outputPath) {
            setOperationResult('Export cancelled: no output folder selected.');
            return;
        }

        let success = 0;
        let failed = 0;
        for (const id of selectedIds) {
            try {
                await exportGenomeWithWeights(id, outputPath);
                success += 1;
            } catch {
                failed += 1;
            }
        }

        setOperationResult(`Export selected: success ${success}, failed ${failed}. Folder: ${outputPath}`);
    };

    const openGenealogy = async (genomeId: string) => {
        setIsLineageLoading(true);
        try {
            const path = await getGenealogyPath(genomeId);
            setLineageRecords(path.records.map((record) => record.genome_id));
        } catch {
            setLineageRecords([]);
        } finally {
            setIsLineageLoading(false);
        }
    };

    return (
        <>
            <TitleBar />
            <div className={styles.container}>
                <div className={styles.headerRow}>
                    <div>
                        <h1 className={styles.title}>Hidden Archive</h1>
                        <p className={styles.subtitle}>Review autosaved hidden genomes with metrics, lineage and batch operations.</p>
                    </div>
                    <div className={styles.topActions}>
                        <button className={styles.secondaryBtn} onClick={() => navigate('/genome-library')}>Back to Library</button>
                        <button className={styles.secondaryBtn} onClick={loadEntries}><BsArrowClockwise /> Refresh</button>
                    </div>
                </div>

                <div className={styles.filtersCard}>
                    <div className={styles.filterGrid}>
                        <label>
                            Search genome id
                            <div className={styles.searchInputWrap}>
                                <BsSearch />
                                <input
                                    value={searchId}
                                    onChange={(e) => setSearchId(e.target.value)}
                                    placeholder="id or name"
                                />
                            </div>
                        </label>

                        <label>
                            Generation min
                            <input value={generationMin} onChange={(e) => setGenerationMin(e.target.value)} placeholder="0" />
                        </label>
                        <label>
                            Generation max
                            <input value={generationMax} onChange={(e) => setGenerationMax(e.target.value)} placeholder="100" />
                        </label>
                        <label>
                            Accuracy min
                            <input value={accuracyMin} onChange={(e) => setAccuracyMin(e.target.value)} placeholder="0.0" />
                        </label>
                        <label>
                            Accuracy max
                            <input value={accuracyMax} onChange={(e) => setAccuracyMax(e.target.value)} placeholder="1.0" />
                        </label>
                        <label>
                            Latency min, ms
                            <input value={latencyMin} onChange={(e) => setLatencyMin(e.target.value)} placeholder="0" />
                        </label>
                        <label>
                            Latency max, ms
                            <input value={latencyMax} onChange={(e) => setLatencyMax(e.target.value)} placeholder="1000" />
                        </label>
                        <label>
                            Sort
                            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
                                <option value="created_desc">Created desc</option>
                                <option value="generation_desc">Generation desc</option>
                                <option value="accuracy_desc">Accuracy desc</option>
                                <option value="latency_asc">Latency asc</option>
                                <option value="model_size_asc">Model size asc</option>
                            </select>
                        </label>
                    </div>

                    <div className={styles.filterActions}>
                        <button className={styles.secondaryBtn} onClick={loadEntries}>Apply backend filters</button>
                    </div>
                </div>

                <div className={styles.batchBar}>
                    <span>{selectedCount} selected</span>
                    <div className={styles.batchActions}>
                        <button className={styles.secondaryBtn} onClick={handleBatchUnhide}>Unhide selected</button>
                        <button className={styles.dangerBtn} onClick={handleBatchDelete}><BsTrash /> Delete selected</button>
                        <button className={styles.primaryBtn} onClick={handleBatchExport}><BsBoxArrowUpRight /> Export selected</button>
                    </div>
                </div>

                {operationResult && <div className={styles.operationResult}>{operationResult}</div>}
                {error && <div className={styles.error}>{error}</div>}

                <div className={styles.tableWrap}>
                    <table className={styles.table}>
                        <thead>
                            <tr>
                                <th>
                                    <input
                                        aria-label="Select all visible"
                                        type="checkbox"
                                        checked={allVisibleSelected}
                                        onChange={toggleSelectAllVisible}
                                    />
                                </th>
                                <th>genome_id</th>
                                <th>generation</th>
                                <th>accuracy</th>
                                <th>latency</th>
                                <th>model_size</th>
                                <th>created_at</th>
                                <th>parents_count</th>
                                <th>actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {!isLoading && visibleEntries.length === 0 && (
                                <tr>
                                    <td colSpan={9} className={styles.emptyState}>
                                        <BsArchive />
                                        <span>No hidden genomes found</span>
                                    </td>
                                </tr>
                            )}
                            {isLoading && (
                                <tr>
                                    <td colSpan={9} className={styles.emptyState}>Loading hidden archive...</td>
                                </tr>
                            )}
                            {!isLoading && visibleEntries.map((entry) => {
                                const isSelected = selectedIds.has(entry.id);
                                return (
                                    <tr key={entry.id} className={isSelected ? styles.selectedRow : ''}>
                                        <td>
                                            <input
                                                aria-label={`Select ${entry.id}`}
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleSelectOne(entry.id)}
                                            />
                                        </td>
                                        <td>{entry.id}</td>
                                        <td>{entry.sourceGeneration ?? 0}</td>
                                        <td>{metricAccuracy(entry).toFixed(4)}</td>
                                        <td>{Number.isFinite(metricLatency(entry)) ? metricLatency(entry).toFixed(2) : 'N/A'}</td>
                                        <td>{Number.isFinite(metricModelSize(entry)) ? metricModelSize(entry).toFixed(3) : 'N/A'}</td>
                                        <td>{new Date(entry.createdAt).toLocaleString()}</td>
                                        <td>{entry.parentGenomes?.length ?? 0}</td>
                                        <td>
                                            <button
                                                className={styles.tableActionBtn}
                                                onClick={() => {
                                                    setActiveDetailId(entry.id);
                                                    setLineageRecords([]);
                                                }}
                                            >
                                                <BsInfoCircle /> Details
                                            </button>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {activeDetail && (
                <div className={styles.modalOverlay} onClick={() => setActiveDetailId(null)}>
                    <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
                        <div className={styles.modalHeader}>
                            <h2>Hidden Genome: {activeDetail.id}</h2>
                            <button className={styles.secondaryBtn} onClick={() => setActiveDetailId(null)}>Close</button>
                        </div>

                        <div className={styles.modalGrid}>
                            <section className={styles.modalCard}>
                                <h3>Objectives</h3>
                                <p>Generation: {activeDetail.sourceGeneration ?? 0}</p>
                                <p>Accuracy: {metricAccuracy(activeDetail).toFixed(4)}</p>
                                <p>Latency: {Number.isFinite(metricLatency(activeDetail)) ? `${metricLatency(activeDetail).toFixed(2)} ms` : 'N/A'}</p>
                                <p>Model size: {Number.isFinite(metricModelSize(activeDetail)) ? `${metricModelSize(activeDetail).toFixed(3)} MB` : 'N/A'}</p>
                            </section>

                            <section className={styles.modalCard}>
                                <h3>Profiler breakdown</h3>
                                {activeDetail.profilerData ? (
                                    <>
                                        <p>Train: {(activeDetail.profilerData.total_train_duration_ms / 1000).toFixed(2)} s</p>
                                        <p>Inference/sample: {activeDetail.profilerData.inference_msec_per_sample.toFixed(3)} ms</p>
                                        <p>Peak memory: {activeDetail.profilerData.peak_active_memory_mb.toFixed(2)} MB</p>
                                        <p>Throughput: {activeDetail.profilerData.samples_per_sec.toFixed(1)} samples/s</p>
                                    </>
                                ) : (
                                    <p>Profiler data is not available.</p>
                                )}
                            </section>

                            <section className={styles.modalCard}>
                                <h3>Lineage summary</h3>
                                <p>Parents: {(activeDetail.parentGenomes ?? []).join(', ') || 'None'}</p>
                                {isLineageLoading && <p>Loading genealogy path...</p>}
                                {!isLineageLoading && lineageRecords.length > 0 && (
                                    <p>Path: {lineageRecords.join(' -> ')}</p>
                                )}
                            </section>
                        </div>

                        <div className={styles.modalActions}>
                            <button className={styles.secondaryBtn} onClick={() => openGenealogy(activeDetail.id)}>Open genealogy</button>
                            <button className={styles.primaryBtn} onClick={() => setExportGenomeId(activeDetail.id)}>Export weights</button>
                        </div>
                    </div>
                </div>
            )}

            {exportGenomeId && (
                <ExportGenomeWithWeightsModal
                    genomeId={exportGenomeId}
                    onClose={() => setExportGenomeId(null)}
                />
            )}
        </>
    );
};
