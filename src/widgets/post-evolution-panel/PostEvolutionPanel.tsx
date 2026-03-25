import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { PopulatedGenome } from '../../entities/genome';
import type { DeviceConstraintParams } from '../../features/evolution-manager';
import { useGenomeLibraryStore } from '../../features/genome-library';
import type {
    GenerationParetoFront,
    GenomeGenealogy,
    GenomeObjectives,
    StoppingCriterionType,
} from '../../shared/lib';
import { GenealogicTreeView } from '../genealogy-tree-viewer';
import { ParetoFrontVisualizer } from '../pareto-front-visualizer';
import {
    buildComparisonRows,
    buildEvolutionReportDataModel,
    buildHiddenArchiveSummary,
    buildLineageExport,
    buildParetoExportPayload,
} from './model';
import styles from './PostEvolutionPanel.module.css';

type Props = {
    paretoHistory: Map<number, GenerationParetoFront>;
    genealogyTree?: Map<string, GenomeGenealogy>;
    onSyncGenealogyTree?: (tree: Map<string, GenomeGenealogy>) => void;
    onOpenGenomeDetails?: (genomeId: string) => void;
    onExportWeights?: (genomeId: string) => void;
    onContinueEvolution?: () => void;
    generation: number;
    elapsedRuntimeSeconds: number;
    stoppingPolicy: StoppingCriterionType[];
    stoppingReason: string;
    genomeById: Map<string, PopulatedGenome>;
    activeDeviceConstraints?: DeviceConstraintParams;
    feasibilityByGenomeId?: Record<string, boolean>;
    constraintViolationScoreByGenomeId?: Record<string, number>;
};

function triggerDownload(filename: string, content: string, type: string) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

function formatRatio(value: number): string {
    if (!Number.isFinite(value)) {
        return 'N/A';
    }
    return `${(value * 100).toFixed(1)}%`;
}

function summarizeStoppingReason(reason: string): string {
    return reason.trim().length > 0 ? reason : 'Stopped manually or by backend event.';
}

export function PostEvolutionPanel({
    paretoHistory,
    genealogyTree,
    onSyncGenealogyTree,
    onOpenGenomeDetails,
    onExportWeights,
    onContinueEvolution,
    generation,
    elapsedRuntimeSeconds,
    stoppingPolicy,
    stoppingReason,
    genomeById,
    activeDeviceConstraints,
    feasibilityByGenomeId = {},
    constraintViolationScoreByGenomeId,
}: Props) {
    const navigate = useNavigate();
    const listHiddenLibrary = useGenomeLibraryStore((state) => state.listHiddenLibrary);

    const [showOnlyFeasible, setShowOnlyFeasible] = useState(false);
    const [selectedGenomeIds, setSelectedGenomeIds] = useState<string[]>([]);
    const [hiddenCount, setHiddenCount] = useState(0);
    const [hiddenLoadError, setHiddenLoadError] = useState<string | null>(null);

    const latestGeneration = useMemo(() => {
        if (paretoHistory.size === 0) {
            return 0;
        }
        return Math.max(...paretoHistory.keys());
    }, [paretoHistory]);

    const latestFront = useMemo<GenerationParetoFront>(() => {
        const fallback: GenerationParetoFront = {
            generation: latestGeneration,
            total_genomes: 0,
            pareto_members: [],
            objectives_3d: [],
            all_genomes: [],
            frontier_genome_ids: [],
        };
        return paretoHistory.get(latestGeneration) ?? fallback;
    }, [latestGeneration, paretoHistory]);

    const allObjectives = useMemo(
        () => latestFront.all_genomes ?? latestFront.pareto_members,
        [latestFront],
    );

    const objectivesByGenomeId = useMemo(() => {
        const next = new Map<string, GenomeObjectives>();
        for (const objective of allObjectives) {
            next.set(objective.genome_id, objective);
        }
        return next;
    }, [allObjectives]);

    const selectableGenomeIds = useMemo(() => {
        const sorted = [...allObjectives].sort((a, b) => b.accuracy - a.accuracy);
        return sorted.map((item) => item.genome_id);
    }, [allObjectives]);

    useEffect(() => {
        if (selectedGenomeIds.length > 0) {
            return;
        }

        setSelectedGenomeIds(selectableGenomeIds.slice(0, Math.min(2, selectableGenomeIds.length)));
    }, [selectableGenomeIds, selectedGenomeIds.length]);

    useEffect(() => {
        let active = true;

        const loadHiddenStats = async () => {
            setHiddenLoadError(null);
            try {
                const entries = await listHiddenLibrary();
                if (active) {
                    setHiddenCount(entries.length);
                }
            } catch (error) {
                if (active) {
                    setHiddenLoadError(String(error));
                }
            }
        };

        loadHiddenStats();
        return () => {
            active = false;
        };
    }, [listHiddenLibrary]);

    const hiddenSummary = useMemo(
        () => buildHiddenArchiveSummary(allObjectives, feasibilityByGenomeId),
        [allObjectives, feasibilityByGenomeId],
    );

    const comparisonRows = useMemo(
        () =>
            buildComparisonRows({
                selectedGenomeIds,
                objectivesByGenomeId,
                genomeById,
                genealogyTree,
                activeDeviceConstraints,
            }),
        [
            activeDeviceConstraints,
            genealogyTree,
            genomeById,
            objectivesByGenomeId,
            selectedGenomeIds,
        ],
    );

    const mutationTimeline = useMemo(() => {
        if (!genealogyTree || genealogyTree.size === 0) {
            return [];
        }

        return [...genealogyTree.values()]
            .sort((a, b) => a.generation - b.generation)
            .slice(-16);
    }, [genealogyTree]);

    const handleSelectGenome = (genomeId: string) => {
        setSelectedGenomeIds((prev) => {
            if (prev.includes(genomeId)) {
                return prev.filter((id) => id !== genomeId);
            }
            if (prev.length >= 3) {
                return [...prev.slice(1), genomeId];
            }
            return [...prev, genomeId];
        });
    };

    const handleDownloadPareto = () => {
        const payload = buildParetoExportPayload(latestFront);
        triggerDownload(`pareto-front-gen-${latestFront.generation}.json`, payload, 'application/json');
    };

    const handleExportLineage = (format: 'json' | 'graphml') => {
        const payload = buildLineageExport(genealogyTree, format);
        const extension = format === 'json' ? 'json' : 'graphml';
        const mime = format === 'json' ? 'application/json' : 'application/graphml+xml';
        triggerDownload(`lineage-gen-${generation}.${extension}`, payload, mime);
    };

    const handleSavePdf = () => {
        const report = buildEvolutionReportDataModel({
            generation,
            elapsedRuntimeSeconds,
            stoppingPolicy,
            stoppingReason: summarizeStoppingReason(stoppingReason),
            paretoFront: latestFront,
            constraints: activeDeviceConstraints,
            hiddenArchive: {
                ...hiddenSummary,
                total: hiddenCount,
            },
        });

        const printable = window.open('', '_blank', 'noopener,noreferrer,width=960,height=720');
        if (!printable) {
            return;
        }

        printable.document.write(`
            <html>
                <head>
                    <title>Evolution Report</title>
                    <style>
                        body { font-family: Segoe UI, sans-serif; margin: 24px; color: #1f2937; }
                        h1 { margin-bottom: 8px; }
                        pre { white-space: pre-wrap; background: #f5f7fb; border: 1px solid #dbe2ee; padding: 12px; border-radius: 8px; }
                    </style>
                </head>
                <body>
                    <h1>Evolution Report</h1>
                    <pre>${JSON.stringify(report, null, 2)}</pre>
                </body>
            </html>
        `);
        printable.document.close();
        printable.focus();
        printable.print();
    };

    const selectedForWeights = selectedGenomeIds[0] ?? selectableGenomeIds[0];

    return (
        <section className={styles.panel} data-testid="post-evolution-panel">
            <div className={styles.headerRow}>
                <div>
                    <h2 className={styles.title}>Post-Evolution Analysis</h2>
                    <p className={styles.subtitle}>Pareto selection, lineage inspection and export actions from one panel.</p>
                </div>
                <label className={styles.feasibleToggle}>
                    <input
                        type="checkbox"
                        checked={showOnlyFeasible}
                        onChange={(event) => setShowOnlyFeasible(event.target.checked)}
                    />
                    Show only feasible for selected device
                </label>
            </div>

            <div className={styles.legendRow}>
                <span className={styles.legendBadge} data-testid="legend-feasible">
                    <span className={`${styles.legendDot} ${styles.legendDotFeasible}`} />
                    feasible
                </span>
                <span className={styles.legendBadge} data-testid="legend-infeasible">
                    <span className={`${styles.legendDot} ${styles.legendDotInfeasible}`} />
                    infeasible
                </span>
            </div>

            <div className={styles.section}>
                <h3 className={styles.sectionTitle}>Pareto visualization</h3>
                <ParetoFrontVisualizer
                    currentParetoFront={latestFront.pareto_members}
                    paretoHistory={paretoHistory}
                    feasibilityByGenomeId={feasibilityByGenomeId}
                    constraintViolationScoreByGenomeId={constraintViolationScoreByGenomeId}
                    showOnlyFeasible={showOnlyFeasible}
                    onOpenDetails={onOpenGenomeDetails}
                    onExportSelected={onExportWeights}
                />
            </div>

            <div className={styles.section}>
                <h3 className={styles.sectionTitle}>Genome comparison (2-3 genomes)</h3>
                <div className={styles.compareControls}>
                    <select
                        className={styles.compareSelect}
                        value=""
                        onChange={(event) => {
                            if (event.target.value) {
                                handleSelectGenome(event.target.value);
                                event.target.value = '';
                            }
                        }}
                        data-testid="comparison-select"
                    >
                        <option value="">Add genome to comparison...</option>
                        {selectableGenomeIds.map((id) => (
                            <option key={id} value={id} disabled={selectedGenomeIds.includes(id)}>
                                {id}
                            </option>
                        ))}
                    </select>
                    <div className={styles.compareHint}>Selected: {selectedGenomeIds.join(', ') || 'none'}</div>
                </div>

                <div className={styles.tableWrap}>
                    <table className={styles.table}>
                        <thead>
                            <tr>
                                <th>Genome</th>
                                <th>Architecture</th>
                                <th>Accuracy / Latency / Size</th>
                                <th>Train / Infer</th>
                                <th>Memory breakdown</th>
                                <th>Lineage depth</th>
                                <th>Device ratios</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comparisonRows.map((row) => (
                                <tr key={row.genomeId}>
                                    <td>{row.genomeId}</td>
                                    <td>{row.architectureSummary}</td>
                                    <td>
                                        {row.accuracy.toFixed(4)} / {row.latencyMs.toFixed(3)} ms / {row.modelSizeMb.toFixed(3)} MB
                                    </td>
                                    <td>{row.trainingTimeMs.toFixed(1)} ms / {row.inferenceTimeMs.toFixed(3)} ms</td>
                                    <td>{row.memoryBreakdown}</td>
                                    <td>{row.lineageDepth}</td>
                                    <td>
                                        <div>MOPS: {formatRatio(row.deviceRatios.mops)}</div>
                                        <div>RAM: {formatRatio(row.deviceRatios.ram)}</div>
                                        <div>FLASH: {formatRatio(row.deviceRatios.flash)}</div>
                                        <div>LAT: {formatRatio(row.deviceRatios.latency)}</div>
                                    </td>
                                    <td>
                                        <button
                                            type="button"
                                            className={styles.button}
                                            onClick={() => onOpenGenomeDetails?.(row.genomeId)}
                                        >
                                            Open
                                        </button>
                                    </td>
                                </tr>
                            ))}
                            {comparisonRows.length === 0 && (
                                <tr>
                                    <td colSpan={8} className={styles.small}>Add genomes to start side-by-side comparison.</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className={styles.section}>
                <h3 className={styles.sectionTitle}>Genealogy analysis</h3>
                <GenealogicTreeView
                    genealogyTree={genealogyTree}
                    paretoHistory={paretoHistory}
                    onOpenGenomeDetails={onOpenGenomeDetails}
                    onGenealogyTreeSync={onSyncGenealogyTree}
                />
                <div className={styles.timeline}>
                    {mutationTimeline.map((item) => (
                        <div
                            key={`${item.genome_id}-${item.generation}`}
                            className={styles.timelineItem}
                            title={`Mutation: ${JSON.stringify(item.mutation_type)}`}
                        >
                            <div>G{item.generation} {'->'} {item.genome_id}</div>
                            <div>fitness {item.fitness.toFixed(4)}</div>
                        </div>
                    ))}
                </div>
                <div className={styles.actionsRow}>
                    <button type="button" className={styles.button} onClick={() => handleExportLineage('json')}>
                        Export lineage JSON
                    </button>
                    <button type="button" className={styles.button} onClick={() => handleExportLineage('graphml')}>
                        Export lineage GraphML
                    </button>
                </div>
            </div>

            <div className={styles.section}>
                <h3 className={styles.sectionTitle}>Hidden archive summary</h3>
                <div>N genomes auto-saved: {hiddenCount}</div>
                <div>Avg fitness: {hiddenSummary.avgFitness.toFixed(4)}</div>
                <div>Accuracy range: {hiddenSummary.minAccuracy.toFixed(4)} - {hiddenSummary.maxAccuracy.toFixed(4)}</div>
                <div>Feasible count: {hiddenSummary.feasibleCount}</div>
                {hiddenLoadError && <div className={styles.error}>Hidden archive load issue: {hiddenLoadError}</div>}
                <div className={styles.actionsRow}>
                    <button type="button" className={styles.button} onClick={() => navigate('/hidden-archive')}>
                        Open Hidden Archive
                    </button>
                </div>
            </div>

            <div className={styles.section}>
                <h3 className={styles.sectionTitle}>Export actions</h3>
                <div className={styles.actionsRow}>
                    <button type="button" className={styles.button} onClick={handleDownloadPareto}>
                        Download Pareto Front (JSON)
                    </button>
                    <button
                        type="button"
                        className={styles.button}
                        onClick={() => selectedForWeights && onExportWeights?.(selectedForWeights)}
                        disabled={!selectedForWeights}
                    >
                        Select and Export Model Weights
                    </button>
                    <button type="button" className={styles.button} onClick={handleSavePdf}>
                        Save Evolution Report (PDF)
                    </button>
                    <button type="button" className={`${styles.button} ${styles.primary}`} onClick={onContinueEvolution}>
                        Continue Evolution
                    </button>
                </div>
            </div>
        </section>
    );
}
