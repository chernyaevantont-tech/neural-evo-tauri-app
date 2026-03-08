import React, { useState, useEffect, useMemo } from 'react';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import styles from './EvolutionStudioPage.module.css';
import { useNavigate } from 'react-router-dom';
import { BsArrowLeft, BsPlay, BsStop, BsPlus, BsX } from 'react-icons/bs';
import { useEvolutionLoop } from '../../features/evolution-studio/model/useEvolutionLoop';
import { useDatasetManagerStore } from '../../features/dataset-manager/model/store';
import { useCanvasGenomeStore, serializeGenome } from '../../entities/canvas-genome';
import { GenomeCatalogPicker } from '../../features/genome-library';
import { useGenomeLibraryStore } from '../../features/genome-library/model/store';
import { EvolutionSettingsPanel } from './EvolutionSettingsPanel';
import { PopulatedGenome, GenerationSnapshot } from '../../features/evolution-studio/model/useEvolutionLoop';
import { GenomeSvgPreview } from '../../entities/canvas-genome/ui/GenomeSvgPreview/GenomeSvgPreview';
import { InspectGenomeModal } from './InspectGenomeModal';
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
    ChartData
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export const EvolutionStudioPage: React.FC = () => {
    const navigate = useNavigate();
    const datasetProfileId = useDatasetManagerStore(state => state.selectedProfileId);
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
    } = useEvolutionLoop(datasetProfileId);

    const genomes = useCanvasGenomeStore(state => state.genomes);
    const { entries, loadGenomeContent } = useGenomeLibraryStore();
    const profiles = useDatasetManagerStore(state => state.profiles);

    const [showCatalogPicker, setShowCatalogPicker] = useState(false);
    const [selectedSeedIds, setSelectedSeedIds] = useState<string[]>([]);
    const [inspectingGenome, setInspectingGenome] = useState<PopulatedGenome | null>(null);

    // Bottom panel state
    const [bottomTab, setBottomTab] = useState<'generations' | 'log'>('generations');
    const [viewingGenIndex, setViewingGenIndex] = useState(0);
    const [autoFollow, setAutoFollow] = useState(true);
    type SortKey = 'evalOrder' | 'accuracy' | 'fitness' | 'nodes';
    const [sortKey, setSortKey] = useState<SortKey>('evalOrder');
    const [sortAsc, setSortAsc] = useState(false);

    // Auto-follow latest generation
    useEffect(() => {
        if (autoFollow && generationHistory.length > 0) {
            setViewingGenIndex(generationHistory.length - 1);
        }
    }, [autoFollow, generationHistory.length]);

    // Sorted genomes for the currently viewed generation
    const viewingSnapshot = generationHistory[viewingGenIndex] as GenerationSnapshot | undefined;
    const sortedGenomes = useMemo(() => {
        if (!viewingSnapshot) return [];
        const genomes = [...viewingSnapshot.genomes];
        if (sortKey === 'evalOrder') {
            return sortAsc ? genomes : [...genomes].reverse();
        }
        const comparator = (a: PopulatedGenome, b: PopulatedGenome) => {
            let valA = 0, valB = 0;
            if (sortKey === 'accuracy') { valA = a.accuracy || 0; valB = b.accuracy || 0; }
            else if (sortKey === 'fitness') { valA = a.adjustedFitness || 0; valB = b.adjustedFitness || 0; }
            else if (sortKey === 'nodes') { valA = a.nodes.length; valB = b.nodes.length; }
            return sortAsc ? valA - valB : valB - valA;
        };
        return genomes.sort(comparator);
    }, [viewingSnapshot, sortKey, sortAsc]);

    const activeProfile = profiles.find(p => p.id === datasetProfileId);

    const handleStart = async () => {
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
            if (activeGenomes.length === 0) {
                alert("Please add seeds from the Library or create a starting architecture in the Sandbox first!");
                return;
            }
            const seedGenome = activeGenomes[0];
            const seedJson = await serializeGenome(seedGenome.genome);
            seedJsonList.push(seedJson);
        }

        if (seedJsonList.length > 0) {
            startEvolution(seedJsonList);
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

    return (
        <div className={styles.pageContainer}>
            <TitleBar />

            <div className={styles.header}>
                <button className={styles.backButton} onClick={() => navigate('/')}>
                    <BsArrowLeft /> Back to Home
                </button>
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

                    {/* Chart Area */}
                    <div className={styles.chartArea}>
                        <h3 className={styles.sectionTitle}>Fitness Over Time</h3>
                        <div className={styles.chartContainer}>
                            {stats.length > 0 ? (
                                <svg width="100%" height="200" viewBox="0 0 100 100" preserveAspectRatio="none">
                                    {/* Simple Grid/Axes */}
                                    <line x1="0" y1="100" x2="100" y2="100" stroke="var(--color-border-primary)" strokeWidth="1" />
                                    <line x1="0" y1="0" x2="0" y2="100" stroke="var(--color-border-primary)" strokeWidth="1" />
                                    <polyline
                                        fill="none"
                                        stroke="var(--color-accent-primary)"
                                        strokeWidth="2"
                                        points={(() => {
                                            const minGen = 0;
                                            const maxGen = Math.max(...stats.map(s => s.generation), 1);
                                            // Handle case where fitness might be negative or low
                                            const minFitness = Math.min(0, ...stats.map(s => s.bestFitness));
                                            const maxFitness = Math.max(1, ...stats.map(s => s.bestFitness));
                                            const rangeX = maxGen - minGen;
                                            const rangeY = (maxFitness - minFitness) || 1;

                                            return stats.map(s => {
                                                const x = ((s.generation - minGen) / rangeX) * 100;
                                                const y = 100 - (((s.bestFitness - minFitness) / rangeY) * 100);
                                                return `${x},${y}`;
                                            }).join(' ');
                                        })()}
                                    />
                                </svg>
                            ) : (
                                <div className={styles.chartPlaceholder}>
                                    Run evolution to see fitness chart
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Bottom Tabbed Panel: Generations / Event Log */}
                    <div className={styles.logArea}>
                        <div className={styles.tabBar}>
                            <button
                                className={`${styles.tabBtn} ${bottomTab === 'generations' ? styles.tabBtnActive : ''}`}
                                onClick={() => setBottomTab('generations')}
                            >
                                Generations {generationHistory.length > 0 ? `(${generationHistory.length})` : ''}
                            </button>
                            <button
                                className={`${styles.tabBtn} ${bottomTab === 'log' ? styles.tabBtnActive : ''}`}
                                onClick={() => setBottomTab('log')}
                            >
                                Event Log
                            </button>
                        </div>

                        {bottomTab === 'log' && (
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
                        )}

                        {bottomTab === 'generations' && (
                            <div className={styles.generationsPanel}>
                                {/* Pagination + Controls */}
                                <div className={styles.genPaginationBar}>
                                    <button
                                        className={styles.genPageBtn}
                                        disabled={viewingGenIndex <= 0}
                                        onClick={() => { setAutoFollow(false); setViewingGenIndex(i => Math.max(0, i - 1)); }}
                                    >←</button>
                                    <span className={styles.genPageLabel}>
                                        Generation {viewingSnapshot ? viewingSnapshot.generation : '--'}
                                        {viewingSnapshot && ` (${viewingSnapshot.timestamp})`}
                                    </span>
                                    <button
                                        className={styles.genPageBtn}
                                        disabled={viewingGenIndex >= generationHistory.length - 1}
                                        onClick={() => { setAutoFollow(false); setViewingGenIndex(i => Math.min(generationHistory.length - 1, i + 1)); }}
                                    >→</button>
                                    <label className={styles.autoFollowLabel}>
                                        <input
                                            type="checkbox"
                                            checked={autoFollow}
                                            onChange={e => setAutoFollow(e.target.checked)}
                                        />
                                        Auto
                                    </label>
                                    <select
                                        className={styles.sortSelect}
                                        value={sortKey}
                                        onChange={e => setSortKey(e.target.value as SortKey)}
                                    >
                                        <option value="evalOrder">Eval Order</option>
                                        <option value="accuracy">Accuracy</option>
                                        <option value="fitness">Fitness</option>
                                        <option value="nodes">Nodes</option>
                                    </select>
                                    <button
                                        className={styles.sortDirBtn}
                                        onClick={() => setSortAsc(v => !v)}
                                        title={sortAsc ? 'Ascending' : 'Descending'}
                                    >{sortAsc ? '↑' : '↓'}</button>
                                </div>

                                {/* Genome Table */}
                                {sortedGenomes.length > 0 ? (
                                    <div className={styles.genomeTableWrap}>
                                        <table className={styles.genomeTable}>
                                            <thead>
                                                <tr>
                                                    <th>#</th>
                                                    <th>Fitness</th>
                                                    <th>Loss</th>
                                                    <th>Acc %</th>
                                                    <th>Nodes</th>
                                                    <th>Flash</th>
                                                    <th>RAM</th>
                                                    <th>MACs</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedGenomes.map((g, idx) => (
                                                    <tr
                                                        key={g.id}
                                                        className={styles.genomeRow}
                                                        onClick={() => setInspectingGenome(g)}
                                                    >
                                                        <td>{idx + 1}</td>
                                                        <td>{g.adjustedFitness?.toFixed(4) || '--'}</td>
                                                        <td>{g.loss?.toFixed(4) || '--'}</td>
                                                        <td>{g.accuracy?.toFixed(2) || '--'}</td>
                                                        <td>{g.nodes.length}</td>
                                                        <td>{g.resources ? (g.resources.totalFlash / 1024).toFixed(1) + 'K' : '--'}</td>
                                                        <td>{g.resources ? (g.resources.totalRam / 1024).toFixed(1) + 'K' : '--'}</td>
                                                        <td>{g.resources ? (g.resources.totalMacs / 1000).toFixed(1) + 'K' : '--'}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <div className={styles.chartPlaceholder}>
                                        No generations evaluated yet.
                                    </div>
                                )}
                            </div>
                        )}
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
                    trainingMetrics={inspectingGenome.trainingMetrics}
                    onClose={() => setInspectingGenome(null)}
                />
            )}
        </div>
    );
};
