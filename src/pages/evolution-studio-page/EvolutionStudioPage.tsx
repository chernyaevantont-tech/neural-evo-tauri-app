import React, { useState, useEffect, useMemo } from 'react';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import styles from './EvolutionStudioPage.module.css';
import { useNavigate } from 'react-router-dom';
import { BsArrowLeft, BsPlay, BsStop, BsPlus, BsX } from 'react-icons/bs';
import { useEvolutionLoop } from '../../features/evolution-studio';
import { useEvolutionSettingsStore } from '../../features/evolution-manager';
import { useDatasetManagerStore } from '../../features/dataset-manager';
import { useCanvasGenomeStore, serializeGenome } from '../../entities/canvas-genome';
import type { GenerationSnapshot, PopulatedGenome } from '../../entities/genome';
import { GenomeCatalogPicker } from '../../features/genome-library';
import { useGenomeLibraryStore } from '../../features/genome-library';
import { GenomeDetailPanel } from '../../features/genome-library/ui/GenomeDetailPanel';
import { useProfilerStats } from '../../shared/hooks';
import { EvolutionSettingsPanel } from './EvolutionSettingsPanel';
import { GenomeSvgPreview } from '../../entities/canvas-genome/ui/GenomeSvgPreview/GenomeSvgPreview';
import { InspectGenomeModal } from './InspectGenomeModal';
import { GenomeProfilerModal } from '../../features/evolution-studio/ui/GenomeProfilerModal';
import { GenerationStatsTable } from '../../features/evolution-studio/ui/GenerationStatsTable';
import { ComparisonCharts } from '../../widgets/genome-comparison/ComparisonCharts';
import { ParetoFrontVisualizer } from '../../widgets/pareto-front-visualizer';
import { GenerationsModal } from './GenerationsModal';
import type { GenerationParetoFront, GenomeObjectives } from '../../shared/lib';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ChartOptions,
    ChartData,
    Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

function isDominatedBy(a: GenomeObjectives, b: GenomeObjectives): boolean {
    const strictBetter =
        b.accuracy > a.accuracy ||
        b.inference_latency_ms < a.inference_latency_ms ||
        b.model_size_mb < a.model_size_mb;

    return (
        b.accuracy >= a.accuracy &&
        b.inference_latency_ms <= a.inference_latency_ms &&
        b.model_size_mb <= a.model_size_mb &&
        strictBetter
    );
}

function computeParetoMembers(objectives: GenomeObjectives[]): GenomeObjectives[] {
    return objectives.filter((candidate) => {
        return !objectives.some(
            (other) => other.genome_id !== candidate.genome_id && isDominatedBy(candidate, other),
        );
    });
}

function mapToObjectives(genome: PopulatedGenome): GenomeObjectives {
    const totalFlashBytes = genome.resources?.totalFlash ?? 0;
    const modelSizeMb = totalFlashBytes / (1024 * 1024);

    return {
        genome_id: genome.id,
        accuracy: genome.accuracy ?? 0,
        inference_latency_ms: genome.profiler?.inference_msec_per_sample ?? 0,
        model_size_mb: modelSizeMb,
        training_time_ms: genome.profiler?.total_train_duration_ms ?? 0,
        is_dominated: false,
        domination_count: 0,
    };
}

export const EvolutionStudioPage: React.FC = () => {
    const navigate = useNavigate();
    const datasetProfileId = useDatasetManagerStore(state => state.selectedProfileId);
    const profiles = useDatasetManagerStore(state => state.profiles);
    const settings = useEvolutionSettingsStore();
    const {
        isRunning,
        startEvolution,
        stopEvolution,
        generation,
        population,
        hallOfFame,
        stats,
        logs,
        runGeneration,
        currentEvaluatingIndex,
        liveMetrics,
        generationHistory
    } = useEvolutionLoop({
        datasetProfileId,
        settings,
        datasetProfiles: profiles,
    });

    const genomes = useCanvasGenomeStore(state => state.genomes);
    const { entries, loadGenomeContent } = useGenomeLibraryStore();

    const [showCatalogPicker, setShowCatalogPicker] = useState(false);
    const [selectedSeedIds, setSelectedSeedIds] = useState<string[]>([]);
    const [paretoSeedJsonByGenomeId, setParetoSeedJsonByGenomeId] = useState<Record<string, string>>({});
    const [inspectingGenome, setInspectingGenome] = useState<PopulatedGenome | null>(null);
    const [profilerGenome, setProfilerGenome] = useState<PopulatedGenome | null>(null);
    const [showGenerationsModal, setShowGenerationsModal] = useState(false);

    // Auto-follow latest generation
    useEffect(() => {
        if (generationHistory.length === 0) {
            return;
        }

        const evaluated = [...generationHistory].reverse().find((s) => s.evaluated);
        if (!evaluated) {
            return;
        }

        const peakConcurrentVramMb = evaluated.genomes.reduce(
            (max, g) => Math.max(max, g.profiler?.peak_active_memory_mb ?? 0),
            0,
        );

        settings.setGenerationProfilingStat(evaluated.generation, {
            generation: evaluated.generation,
            totalTrainingMs: evaluated.totalTrainingMs ?? 0,
            totalInferenceMs: evaluated.totalInferenceMs ?? 0,
            avgSamplesPerSec: evaluated.avgSamplesPerSec ?? 0,
            peakConcurrentVramMb,
            totalJobsCompleted: evaluated.genomes.length,
            totalJobsFailed: 0,
        });
    }, [generationHistory, settings]);

    // Current snapshot (latest generation)
    const currentSnapshot = generationHistory.length > 0 
        ? generationHistory[generationHistory.length - 1] 
        : undefined;
    const profilerStats = useProfilerStats(currentSnapshot?.genomes ?? []);
    const sortedGenomes = useMemo(() => {
        if (!currentSnapshot) return [];
        return [...currentSnapshot.genomes];
    }, [currentSnapshot]);

    const activeProfile = profiles.find(p => p.id === datasetProfileId);
    const genomeById = useMemo(() => {
        const map = new Map<string, PopulatedGenome>();
        for (const genome of population) {
            map.set(genome.id, genome);
        }
        for (const genome of hallOfFame) {
            map.set(genome.id, genome);
        }
        for (const genome of currentSnapshot?.genomes ?? []) {
            map.set(genome.id, genome);
        }
        return map;
    }, [currentSnapshot?.genomes, hallOfFame, population]);

    useEffect(() => {
        if (!currentSnapshot || currentSnapshot.genomes.length === 0) {
            return;
        }

        const allObjectives = currentSnapshot.genomes.map(mapToObjectives);
        const paretoMembers = computeParetoMembers(allObjectives);
        const payload: GenerationParetoFront = {
            generation: currentSnapshot.generation,
            total_genomes: allObjectives.length,
            pareto_members: paretoMembers,
            objectives_3d: paretoMembers.map((item) => [
                item.accuracy,
                item.inference_latency_ms,
                item.model_size_mb,
            ]),
            all_genomes: allObjectives,
            frontier_genome_ids: paretoMembers.map((item) => item.genome_id),
        };

        settings.setParetoForGeneration(currentSnapshot.generation, payload);
    }, [currentSnapshot, settings]);

    const handleUseParetoAsSeed = async (genomeId: string) => {
        const selected = genomeById.get(genomeId);
        if (!selected) {
            return;
        }
        const serialized = await serializeGenome(selected.genome);
        setParetoSeedJsonByGenomeId((prev) => ({
            ...prev,
            [genomeId]: serialized,
        }));
    };

    const handleOpenParetoDetails = (genomeId: string) => {
        const selected = genomeById.get(genomeId);
        if (selected) {
            setInspectingGenome(selected);
        }
    };

    const handleExportParetoSelected = async (genomeId: string) => {
        const selected = genomeById.get(genomeId);
        if (!selected) {
            return;
        }

        const serialized = await serializeGenome(selected.genome);
        const blob = new Blob([serialized], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `genome-${genomeId}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const handleStart = async () => {
        try {
            // First, ensure a dataset profile is selected
            if (!datasetProfileId) {
                alert("Please select a Dataset Profile first!");
                return;
            }

            const seedJsonList: string[] = [];

            if (selectedSeedIds.length > 0) {
                // Load selected seeds from library
                for (const id of selectedSeedIds) {
                    try {
                        const json = await loadGenomeContent(id);
                        seedJsonList.push(json);
                    } catch (e) {
                        console.error("Failed to load seed content for", id, e);
                    }
                }
            } else {
                // Fallback to active sandbox genome
                const activeGenomes = Array.from(genomes.values());
                if (activeGenomes.length > 0) {
                    const seedGenome = activeGenomes[0];
                    const seedJson = await serializeGenome(seedGenome.genome);
                    seedJsonList.push(seedJson);
                }
                // If no seeds and no active genomes, check if random initialization is enabled
                else if (!settings.useRandomInitialization) {
                    alert("Please either:\n1. Add seeds from the Library, OR\n2. Create an architecture in the Sandbox, OR\n3. Enable 'Random Initialization' to generate random architectures");
                    return;
                }
            }

            const paretoSeedJsons = Object.values(paretoSeedJsonByGenomeId);
            if (paretoSeedJsons.length > 0) {
                seedJsonList.push(...paretoSeedJsons);
            }

            // Allow evolution with or without seeds if random initialization is enabled
            if (seedJsonList.length > 0 || settings.useRandomInitialization) {
                startEvolution(seedJsonList);
            }
        } catch (err: any) {
            console.error(err);
            alert("Error in handleStart: " + (err.message || String(err)));
        }
    };

    // Calculate Average Nodes
    const avgNodes = population.length > 0
        ? Math.round(population.reduce((acc, pop) => acc + pop.nodes.length, 0) / population.length)
        : 0;

    // Best Fitness Display
    const bestFitness = hallOfFame.length > 0
        ? hallOfFame[0].adjustedFitness?.toFixed(4) || '--'
        : '--';



    // Chart.js Data and Options
    const liveChartData: ChartData<'line'> = {
        labels: liveMetrics.map(m => m.batch.toString()),
        datasets: [
            {
                label: 'Loss',
                yAxisID: 'y',
                data: liveMetrics.map(m => m.loss),
                borderColor: '#ffb86c',
                backgroundColor: 'rgba(255, 184, 108, 0.15)',
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Accuracy (%)',
                yAxisID: 'y1',
                data: liveMetrics.map(m => m.accuracy),
                borderColor: '#50fa7b',
                backgroundColor: 'rgba(80, 250, 123, 0.15)',
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2,
            }
        ]
    };

    const liveChartOptions: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            x: {
                display: false,
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: { display: true, text: 'Loss', color: '#999' },
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255, 255, 255, 0.08)' }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: { display: true, text: 'Accuracy %', color: '#999' },
                ticks: { color: '#aaa' },
                min: 0,
                max: 100,
                grid: { drawOnChartArea: false },
            }
        },
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#ccc' }
            },
            tooltip: {
                callbacks: {
                    title: (context) => {
                        const idx = context[0].dataIndex;
                        const m = liveMetrics[idx];
                        return `Epoch: ${m.epoch} | Batch: ${m.batch} / ${m.total_batches}`;
                    }
                }
            }
        }
    };

    const fitnessChartData: ChartData<'line'> = {
        labels: stats.map(s => s.generation.toString()),
        datasets: [
            {
                label: 'Best Fitness',
                data: stats.map(s => s.bestFitness),
                borderColor: '#bd93f9',
                backgroundColor: 'rgba(189, 147, 249, 0.15)',
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: '#bd93f9',
                borderWidth: 2,
                fill: true,
            }
        ]
    };

    const fitnessChartOptions: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            x: {
                display: true,
                title: { display: true, text: 'Generation', color: '#999' },
                ticks: { color: '#aaa', maxTicksLimit: 10 },
                grid: { color: 'rgba(255, 255, 255, 0.08)' }
            },
            y: {
                display: true,
                title: { display: true, text: 'Fitness', color: '#999' },
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255, 255, 255, 0.08)' }
            }
        },
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                callbacks: {
                    title: (context) => `Generation: ${context[0].label}`,
                }
            }
        }
    };

    return (
        <div className={styles.pageContainer}>
            <TitleBar />

            <div className={styles.header}>
                <h1 className={styles.title}>Evolution Studio</h1>
                <div className={styles.controls}>
                    <button
                        className={`${styles.actionButton} ${styles.startBtn}`}
                        onClick={handleStart}
                        disabled={isRunning}
                        style={{ opacity: isRunning ? 0.5 : 1, cursor: isRunning ? 'not-allowed' : 'pointer' }}
                    >
                        <BsPlay /> Start Evolution
                    </button>
                    <button
                        className={`${styles.actionButton} ${styles.stopBtn}`}
                        onClick={stopEvolution}
                        disabled={!isRunning}
                        style={{ opacity: !isRunning ? 0.5 : 1, cursor: !isRunning ? 'not-allowed' : 'pointer' }}
                    >
                        <BsStop /> Stop
                    </button>
                </div>
            </div>

            <div className={styles.contentLayout}>
                {/* Left Panel: Evolution Settings */}
                <EvolutionSettingsPanel disabled={isRunning} />

                {/* Main Dashboard Area */}
                <div className={styles.mainArea}>

                    {/* Dashboard Grid */}
                    <div className={styles.dashboardGrid}>
                        {/* Left Column: Config & Simple Stats */}
                        <div className={styles.leftColumn}>
                            <div className={styles.setupCard}>
                                <h3 className={styles.setupTitle}>Dataset Profile</h3>
                                {activeProfile ? (
                                    <div className={styles.datasetInfo}>
                                        <span className={styles.datasetName}>{activeProfile.name}</span>
                                        <span className={styles.datasetType}>{activeProfile.type}</span>
                                    </div>
                                ) : (
                                    <div className={styles.warningAlert}>
                                        No dataset selected. Go to Dataset Manager to configure one.
                                    </div>
                                )}
                            </div>

                            <div className={styles.setupCard}>
                                <h3 className={styles.setupTitle}>Initial Population Seeds</h3>
                                <div className={styles.seedList}>
                                    {selectedSeedIds.map(id => {
                                        const entry = entries.find(e => e.id === id);
                                        return (
                                            <div key={id} className={styles.seedTag}>
                                                {entry ? entry.name : 'Unknown'}
                                                {!isRunning && (
                                                    <button
                                                        className={styles.removeSeedBtn}
                                                        onClick={() => setSelectedSeedIds(prev => prev.filter(s => s !== id))}
                                                    >
                                                        <BsX />
                                                    </button>
                                                )}
                                            </div>
                                        )
                                    })}
                                    {!isRunning && (
                                        <button
                                            className={styles.addSeedBtn}
                                            onClick={() => setShowCatalogPicker(true)}
                                        >
                                            <BsPlus /> Add Seeds from Library
                                        </button>
                                    )}
                                </div>
                                {selectedSeedIds.length === 0 && (
                                    <p className={styles.setupHint}>
                                        If empty, the current Sandbox architecture will be used as a single seed.
                                    </p>
                                )}
                            </div>

                            {/* Metrics Column */}
                            <div className={styles.metricsColumn}>
                                <div className={styles.metricCard}>
                                    <div className={styles.metricLabel}>Gen.</div>
                                    <div className={styles.metricValue}>{generation}</div>
                                </div>
                                <div className={styles.metricCard}>
                                    <div className={styles.metricLabel}>Best Fit</div>
                                    <div className={styles.metricValue}>{bestFitness}</div>
                                </div>
                                <div className={styles.metricCard}>
                                    <div className={styles.metricLabel}>Nodes</div>
                                    <div className={styles.metricValue}>{avgNodes}</div>
                                </div>
                                    <div className={styles.metricCard}>
                                        <div className={styles.metricLabel}>Train Avg</div>
                                        <div className={styles.metricValue}>{(profilerStats.avgTrainingTime / 1000).toFixed(2)}s</div>
                                    </div>
                                    <div className={styles.metricCard}>
                                        <div className={styles.metricLabel}>Infer Avg</div>
                                        <div className={styles.metricValue}>{profilerStats.avgInferenceLatency.toFixed(3)}ms</div>
                                    </div>
                                    <div className={styles.metricCard}>
                                        <div className={styles.metricLabel}>Throughput</div>
                                        <div className={styles.metricValue}>{profilerStats.avgThroughput.toFixed(1)}/s</div>
                                    </div>
                            </div>
                        </div>

                        {/* Right Column: Visualizations (Topology & Live Chart) */}
                        <div className={styles.rightColumn}>
                            {isRunning ? (
                                <div className={styles.setupCard} style={{ flex: 1, padding: 0, overflow: 'hidden', position: 'relative' }}>
                                    <div style={{ position: 'absolute', top: '1rem', left: '1rem', zIndex: 10 }}>
                                        <h3 className={styles.setupTitle}>
                                            Evaluating Genome {currentEvaluatingIndex + 1}/{population.length}
                                        </h3>
                                    </div>
                                    <div style={{ width: '100%', height: '100%' }}>
                                        {population.length > 0 ? (
                                            <GenomeSvgPreview nodes={population[currentEvaluatingIndex < population.length ? currentEvaluatingIndex : 0].nodes} />
                                        ) : (
                                            <div className={styles.chartPlaceholder}>Preparing initial generation...</div>
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className={styles.setupCard} style={{ flex: 1, padding: 0, overflow: 'hidden', position: 'relative' }}>
                                    <div style={{ position: 'absolute', top: '1rem', left: '1rem', zIndex: 10 }}>
                                        <h3 className={styles.setupTitle}>
                                            Generation Best Topology
                                        </h3>
                                    </div>
                                    <div style={{ width: '100%', height: '100%' }}>
                                        {population.length > 0 ? (
                                            <GenomeSvgPreview nodes={population[0].nodes} />
                                        ) : (
                                            <div className={styles.chartPlaceholder}>Run evolution to see champion...</div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Live Genome Metrics Chart (Full Width) */}
                    <div className={styles.liveChartArea}>
                        <h3 className={styles.sectionTitle} style={{ marginBottom: "0.5rem" }}>
                            Live Evaluation Metrics
                            {liveMetrics.length > 0 && (
                                <span style={{ fontSize: '0.8rem', fontWeight: 'normal', marginLeft: '0.8rem', color: 'var(--color-text-muted)' }}>
                                    (Batch {liveMetrics[liveMetrics.length - 1].batch} / {liveMetrics[liveMetrics.length - 1].total_batches})
                                </span>
                            )}
                        </h3>
                        <div className={styles.liveChartContainer} style={{ padding: '1rem' }}>
                            {liveMetrics.length > 0 ? (
                                <Line data={liveChartData} options={liveChartOptions} />
                            ) : (
                                <div className={styles.chartPlaceholder}>
                                    Waiting for evaluation to start...
                                </div>
                            )}
                        </div>
                    </div>

                    <div className={styles.chartArea}>
                        <h3 className={styles.sectionTitle}>Fitness Over Time</h3>
                        <div className={styles.chartContainer} style={{ padding: '1rem' }}>
                            {stats.length > 0 ? (
                                <Line data={fitnessChartData} options={fitnessChartOptions} />
                            ) : (
                                <div className={styles.chartPlaceholder}>
                                    Waiting for first generation training complete
                                </div>
                            )}
                        </div>
                    </div>

                    <div className={styles.chartArea}>
                        <h3 className={styles.sectionTitle}>Profiler Comparisons</h3>
                        <div style={{ padding: '0.2rem 0 0.4rem', color: 'var(--color-text-secondary)', fontSize: '0.82rem' }}>
                            <button 
                                onClick={() => setShowGenerationsModal(true)}
                                className={styles.viewGenerationsBtn}
                            >
                                📊 View All Generations ({generationHistory.length})
                            </button>
                            {currentSnapshot && (
                                <>
                                    Total train {((currentSnapshot.totalTrainingMs ?? 0) / 1000).toFixed(2)}s | 
                                    Avg inference {((currentSnapshot.totalInferenceMs ?? 0) / Math.max(1, currentSnapshot.genomes.length ?? 1)).toFixed(3)}ms | 
                                    Avg throughput {(currentSnapshot.avgSamplesPerSec ?? 0).toFixed(1)} samples/s
                                </>
                            )}
                        </div>
                        <ComparisonCharts genomes={sortedGenomes} />
                    </div>

                    <div className={styles.chartArea}>
                        <ParetoFrontVisualizer
                            currentParetoFront={settings.currentParetoFront}
                            paretoHistory={settings.paretoHistory}
                            onUseAsSeed={handleUseParetoAsSeed}
                            onOpenDetails={handleOpenParetoDetails}
                            onExportSelected={handleExportParetoSelected}
                        />
                        {Object.keys(paretoSeedJsonByGenomeId).length > 0 && (
                            <div style={{ marginTop: '0.7rem', fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                                Queued Pareto seeds: {Object.keys(paretoSeedJsonByGenomeId).length}
                            </div>
                        )}
                    </div>

                    {/* Bottom Tabbed Panel: Generations / Event Log */}
                    {/* Bottom Panel: Event Log */}
                    <div className={styles.logArea}>
                        <div className={styles.logHeader}>
                            <h4 className={styles.logTitle}>Event Log</h4>
                        </div>
                        <div className={styles.logConsole}>
                            {logs.map((log, idx) => (
                                <div key={idx} className={styles.logEntry} style={{
                                    color: log.type === 'error' ? 'var(--color-danger)' :
                                        log.type === 'warn' ? 'var(--color-warning)' :
                                            log.type === 'success' ? 'var(--color-success)' : 'inherit'
                                }}>
                                    <span style={{ opacity: 0.5, marginRight: '8px' }}>[{log.time}]</span>
                                    {log.message}
                                </div>
                            ))}
                            {logs.length === 0 && <div className={styles.logEntry}>[System] Evolution Studio initialized. Waiting for User...</div>}
                        </div>
                    </div>

                </div>

                {/* Right Panel: Hall of Fame */}
                <div className={styles.sidePanel}>
                    <h3 className={styles.sectionTitle}>Hall of Fame</h3>
                    <p className={styles.panelSubtitle}>Top architectures discovered across all generations.</p>

                    <div className={styles.fameList}>
                        {hallOfFame.length === 0 ? (
                            <div className={styles.emptyFame}>
                                No champions discovered yet.
                            </div>
                        ) : (
                            hallOfFame.map((champ, idx) => (
                                <div key={champ.id} className={styles.fameCard}>
                                    <div className={styles.fameRank}>#{idx + 1}</div>
                                    <div className={styles.fameDetails}>
                                        <div className={styles.fameScore}>Fitness: {champ.adjustedFitness?.toFixed(4)}</div>
                                        <div className={styles.fameNodes}>Nodes: {champ.nodes.length}</div>
                                    </div>
                                    <button
                                        className={styles.inspectBtn}
                                        onClick={() => setInspectingGenome(champ)}
                                    >
                                        Inspect
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>

            </div>

            {showCatalogPicker && (
                <GenomeCatalogPicker
                    onClose={() => setShowCatalogPicker(false)}
                    onConfirm={(entries) => {
                        const ids = entries.map(e => e.id);
                        setSelectedSeedIds(prev => Array.from(new Set([...prev, ...ids])));
                        setShowCatalogPicker(false);
                    }}
                    multi={true}
                />
            )}

            {inspectingGenome && (
                <InspectGenomeModal
                    title={inspectingGenome.adjustedFitness !== undefined
                        ? `Genome — Fitness: ${inspectingGenome.adjustedFitness.toFixed(4)}`
                        : `Champion Rank #${hallOfFame.findIndex(g => g.id === inspectingGenome.id) + 1}`
                    }
                    subtitle={`Loss: ${inspectingGenome.loss?.toFixed(4) || '--'} | Acc: ${inspectingGenome.accuracy?.toFixed(2) || '--'}% | Nodes: ${inspectingGenome.nodes.length}${inspectingGenome.resources ? ` | Flash: ${(inspectingGenome.resources.totalFlash / 1024).toFixed(1)}K | RAM: ${(inspectingGenome.resources.totalRam / 1024).toFixed(1)}K` : ''}`}
                    nodes={inspectingGenome.nodes}
                    trainingMetrics={
                        // If inspecting the currently active genome, merge in the live metrics for real-time progress
                        inspectingGenome.id === population[currentEvaluatingIndex]?.id && isRunning
                            ? [...(inspectingGenome.trainingMetrics || []), ...liveMetrics]
                            : inspectingGenome.trainingMetrics
                    }
                    genomeDetail={
                        <GenomeDetailPanel
                            genome={inspectingGenome}
                            onOpenProfiler={(g) => setProfilerGenome(g)}
                        />
                    }
                    onClose={() => setInspectingGenome(null)}
                />
            )}

            {profilerGenome?.profiler && (
                <GenomeProfilerModal
                    genomeId={profilerGenome.id}
                    profiler={profilerGenome.profiler}
                    onClose={() => setProfilerGenome(null)}
                />
            )}

            {showGenerationsModal && (
                <GenerationsModal
                    generations={generationHistory}
                    selectedGeneration={currentSnapshot?.generation}
                    onSelectGeneration={(gen) => {
                        // Just close the modal; the selection doesn't affect current snapshot anymore
                        setShowGenerationsModal(false);
                    }}
                    onClose={() => setShowGenerationsModal(false)}
                />
            )}
        </div>
    );
};
