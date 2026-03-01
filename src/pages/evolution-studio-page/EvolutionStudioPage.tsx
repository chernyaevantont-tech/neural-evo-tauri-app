import React from 'react';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import styles from './EvolutionStudioPage.module.css';
import { useNavigate } from 'react-router-dom';
import { BsArrowLeft, BsPlayFill, BsStopFill, BsPauseFill } from 'react-icons/bs';
import { useEvolutionLoop } from '../../features/evolution-studio/model/useEvolutionLoop';
import { useDatasetManagerStore } from '../../features/dataset-manager/model/store';
import { useCanvasGenomeStore, serializeGenome } from '../../entities/canvas-genome';

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

    const handleStart = async () => {
        const activeGenomes = Array.from(genomes.values());
        if (activeGenomes.length === 0) {
            alert("Please create a starting architecture in the Sandbox first!");
            navigate('/sandbox');
            return;
        }

        // Use the first available genome as the seed
        const seedGenome = activeGenomes[0];
        const seedJson = await serializeGenome(seedGenome.genome);
        startEvolution(seedJson);
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
        </div>
    );
};
