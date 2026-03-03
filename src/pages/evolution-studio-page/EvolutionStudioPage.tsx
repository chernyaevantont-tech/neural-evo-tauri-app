import React, { useState } from 'react';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import styles from './EvolutionStudioPage.module.css';
import { useNavigate } from 'react-router-dom';
import { BsArrowLeft, BsPlayFill, BsStopFill, BsPlus, BsX } from 'react-icons/bs';
import { useEvolutionLoop } from '../../features/evolution-studio/model/useEvolutionLoop';
import { useDatasetManagerStore } from '../../features/dataset-manager/model/store';
import { useCanvasGenomeStore, serializeGenome } from '../../entities/canvas-genome';
import { GenomeCatalogPicker } from '../../features/genome-library';
import { useGenomeLibraryStore } from '../../features/genome-library/model/store';

export const EvolutionStudioPage: React.FC = () => {
    const navigate = useNavigate();
    const selectedProfileId = useDatasetManagerStore(state => state.selectedProfileId);

    const {
        isRunning,
        startEvolution,
        stopEvolution,
        generation,
        population,
        hallOfFame,
        logs,
        stats
    } = useEvolutionLoop(selectedProfileId);

    const genomes = useCanvasGenomeStore(state => state.genomes);
    const { entries, loadGenomeContent } = useGenomeLibraryStore();
    const profiles = useDatasetManagerStore(state => state.profiles);

    const [showCatalogPicker, setShowCatalogPicker] = useState(false);
    const [selectedSeedIds, setSelectedSeedIds] = useState<string[]>([]);

    const activeProfile = profiles.find(p => p.id === selectedProfileId);

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
                        <BsPlayFill /> Start Evolution
                    </button>
                    <button
                        className={`${styles.actionButton} ${styles.stopBtn}`}
                        onClick={stopEvolution}
                        disabled={!isRunning}
                        style={{ opacity: !isRunning ? 0.5 : 1, cursor: !isRunning ? 'not-allowed' : 'pointer' }}
                    >
                        <BsStopFill /> Stop
                    </button>
                </div>
            </div>

            <div className={styles.contentLayout}>
                {/* Main Dashboard Area */}
                <div className={styles.mainArea}>

                    {/* Setup Section */}
                    <div className={styles.setupSection}>
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
                            {selectedSeedIds.length === 0 && !isRunning && (
                                <p className={styles.setupHint}>
                                    If empty, the current Sandbox architecture will be used as a single seed.
                                </p>
                            )}
                        </div>
                    </div>

                    {/* Metrics Strip */}
                    <div className={styles.metricsStrip}>
                        <div className={styles.metricCard}>
                            <div className={styles.metricLabel}>Current Generation</div>
                            <div className={styles.metricValue}>{generation}</div>
                        </div>
                        <div className={styles.metricCard}>
                            <div className={styles.metricLabel}>Best Fitness</div>
                            <div className={styles.metricValue}>{bestFitness}</div>
                        </div>
                        <div className={styles.metricCard}>
                            <div className={styles.metricLabel}>Avg. Population Nodes</div>
                            <div className={styles.metricValue}>{avgNodes}</div>
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

                    {/* Event Log */}
                    <div className={styles.logArea}>
                        <h3 className={styles.sectionTitle}>Evolution Events</h3>
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
                                    <button className={styles.inspectBtn} title="Will load in next phase">Inspect</button>
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
        </div>
    );
};
