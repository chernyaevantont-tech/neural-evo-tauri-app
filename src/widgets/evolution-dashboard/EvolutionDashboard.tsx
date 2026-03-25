import React, { useMemo, useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    type ChartData,
    type ChartOptions,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import type { BatchMetrics, GenerationSnapshot, PopulatedGenome } from '../../entities/genome';
import type {
    GenerationParetoFront,
    GenomeGenealogy,
    GenomeObjectives,
    StoppingCriterionType,
} from '../../shared/lib';
import { GenealogicTreeView } from '../genealogy-tree-viewer';
import { ParetoFrontVisualizer } from '../pareto-front-visualizer';
import { GenerationStatsTable } from '../../features/evolution-studio/ui/GenerationStatsTable';
import { StoppingCriteriaLiveMonitor } from '../../features/evolution-manager/ui/StoppingCriteriaLiveMonitor';
import { StoppingCriteriaSummary } from '../../features/evolution-manager/ui/StoppingCriteriaSummary';
import type { LogEntry } from '../../features/evolution-studio/model/useEvolutionLoop';
import {
    buildFitnessTimeline,
    buildJobs,
    deriveOverviewMetrics,
    formatDuration,
    formatEta,
    formatMemory,
} from './model/dashboardSelectors';
import styles from './EvolutionDashboard.module.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

type DashboardTab = 'pareto' | 'genealogy' | 'stopping' | 'performance';

interface EvolutionDashboardProps {
    isRunning: boolean;
    isPaused: boolean;
    generation: number;
    generationHistory: GenerationSnapshot[];
    liveMetrics: BatchMetrics[];
    currentEvaluatingIndex: number;
    population: PopulatedGenome[];
    logs: LogEntry[];
    elapsedRuntimeSeconds: number;
    useMaxGenerations: boolean;
    maxGenerations: number;
    currentParetoFront: GenomeObjectives[];
    paretoHistory: Map<number, GenerationParetoFront>;
    feasibilityByGenomeId: Record<string, boolean>;
    constraintViolationScoreByGenomeId: Record<string, number>;
    showOnlyFeasible: boolean;
    genealogyTree?: Map<string, GenomeGenealogy>;
    onGenealogyTreeSync?: (tree: Map<string, GenomeGenealogy>) => void;
    onUseAsSeed: (genomeId: string) => void;
    onOpenGenomeDetails: (genomeId: string) => void;
    onExportSelected: (genomeId: string) => void;
    stoppingCriteria: StoppingCriterionType[];
    triggeredCriterionIndex: number | null;
    bestAccuracyNormalized: number;
    onPause: () => Promise<void> | void;
    onResume: () => Promise<void> | void;
    onStop: () => Promise<void> | void;
    onSaveCheckpoint: () => Promise<void> | void;
    onOpenProfiler: (genome: PopulatedGenome) => void;
}

export function EvolutionDashboard(props: EvolutionDashboardProps) {
    const {
        isRunning,
        isPaused,
        generation,
        generationHistory,
        liveMetrics,
        currentEvaluatingIndex,
        population,
        logs,
        elapsedRuntimeSeconds,
        useMaxGenerations,
        maxGenerations,
        currentParetoFront,
        paretoHistory,
        feasibilityByGenomeId,
        constraintViolationScoreByGenomeId,
        showOnlyFeasible,
        genealogyTree,
        onGenealogyTreeSync,
        onUseAsSeed,
        onOpenGenomeDetails,
        onExportSelected,
        stoppingCriteria,
        triggeredCriterionIndex,
        bestAccuracyNormalized,
        onPause,
        onResume,
        onStop,
        onSaveCheckpoint,
        onOpenProfiler,
    } = props;

    const [tab, setTab] = useState<DashboardTab>('pareto');
    const [showRunningOnly, setShowRunningOnly] = useState(false);
    const [showFailedOnly, setShowFailedOnly] = useState(false);
    const [pendingAction, setPendingAction] = useState<'pause' | 'resume' | 'stop' | 'checkpoint' | null>(null);

    const currentSnapshot = generationHistory.length > 0 ? generationHistory[generationHistory.length - 1] : undefined;

    const overview = useMemo(
        () => deriveOverviewMetrics({
            generation,
            generationHistory,
            elapsedTimeSeconds: elapsedRuntimeSeconds,
            useMaxGenerations,
            maxGenerations,
            currentParetoFrontSize: currentParetoFront.length,
            feasibleByGenomeId: feasibilityByGenomeId,
        }),
        [
            generation,
            generationHistory,
            elapsedRuntimeSeconds,
            useMaxGenerations,
            maxGenerations,
            currentParetoFront.length,
            feasibilityByGenomeId,
        ],
    );

    const fitnessTimeline = useMemo(
        () =>
            buildFitnessTimeline(
                generationHistory,
                paretoHistory,
                feasibilityByGenomeId,
                constraintViolationScoreByGenomeId,
            ),
        [generationHistory, paretoHistory, feasibilityByGenomeId, constraintViolationScoreByGenomeId],
    );

    const jobs = useMemo(
        () => buildJobs({
            isRunning,
            currentEvaluatingIndex,
            liveMetrics,
            population,
            currentSnapshot,
        }),
        [isRunning, currentEvaluatingIndex, liveMetrics, population, currentSnapshot],
    );

    const visibleJobs = jobs.filter((job) => {
        if (showRunningOnly && job.status !== 'running') {
            return false;
        }
        if (showFailedOnly && job.status !== 'failed') {
            return false;
        }
        return true;
    });

    const chartData: ChartData<'line'> = {
        labels: fitnessTimeline.map((point) => point.generation.toString()),
        datasets: [
            {
                label: 'Best fitness',
                data: fitnessTimeline.map((point) => point.bestFitness),
                borderColor: '#83e5b6',
                backgroundColor: 'rgba(131, 229, 182, 0.22)',
                tension: 0.22,
                fill: false,
                pointRadius: 2,
                yAxisID: 'y',
            },
            {
                label: 'Average fitness',
                data: fitnessTimeline.map((point) => point.avgFitness),
                borderColor: '#61a2f2',
                backgroundColor: 'rgba(97, 162, 242, 0.26)',
                tension: 0.22,
                fill: true,
                pointRadius: 1,
                yAxisID: 'y',
            },
            {
                label: 'Feasible front size',
                data: fitnessTimeline.map((point) => point.feasibleFrontSize ?? 0),
                borderColor: '#f4c56f',
                backgroundColor: 'rgba(244, 197, 111, 0.16)',
                tension: 0.2,
                fill: false,
                borderDash: [4, 4],
                pointRadius: 1,
                yAxisID: 'y1',
            },
            {
                label: 'Constraint pressure',
                data: fitnessTimeline.map((point) => point.constraintPressure ?? null),
                borderColor: '#ff8d8d',
                backgroundColor: 'rgba(255, 141, 141, 0.14)',
                tension: 0.2,
                fill: false,
                pointRadius: 0,
                yAxisID: 'y2',
            },
        ],
    };

    const chartOptions: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                grid: { color: 'rgba(255, 255, 255, 0.08)' },
                ticks: { color: '#b5c2d0' },
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                grid: { drawOnChartArea: false },
                ticks: { color: '#e4cc95' },
            },
            y2: {
                type: 'linear',
                display: false,
                position: 'right',
            },
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.08)' },
                ticks: { color: '#b5c2d0' },
            },
        },
        plugins: {
            legend: {
                labels: { color: '#dbe8f3' },
            },
        },
    };

    const runControlAction = (
        action: 'pause' | 'resume' | 'stop' | 'checkpoint',
        callback: () => Promise<void> | void,
    ) => {
        setPendingAction(action);

        try {
            const result = callback();
            if (result && typeof (result as Promise<void>).finally === 'function') {
                (result as Promise<void>).finally(() => setPendingAction(null));
            } else {
                setPendingAction(null);
            }
        } catch {
            setPendingAction(null);
        }
    };

    const handleStop = () => {
        if (!window.confirm('Stop evolution now?')) {
            return;
        }

        runControlAction('stop', onStop);
    };

    const handlePause = () => {
        runControlAction('pause', onPause);
    };

    const handleResume = () => {
        runControlAction('resume', onResume);
    };

    const handleSaveCheckpoint = () => {
        runControlAction('checkpoint', onSaveCheckpoint);
    };

    return (
        <section className={styles.dashboard}>
            <div className={styles.controlBar}>
                <button
                    type="button"
                    className={`${styles.controlButton} ${styles.controlWarn}`}
                    onClick={handlePause}
                    disabled={!isRunning || isPaused || pendingAction !== null}
                >
                    {pendingAction === 'pause' ? 'Pausing...' : 'Pause Evolution'}
                </button>
                <button
                    type="button"
                    className={`${styles.controlButton} ${styles.controlPrimary}`}
                    onClick={handleResume}
                    disabled={!isRunning || !isPaused || pendingAction !== null}
                >
                    {pendingAction === 'resume' ? 'Resuming...' : 'Resume Evolution'}
                </button>
                <button
                    type="button"
                    className={`${styles.controlButton} ${styles.controlDanger}`}
                    onClick={handleStop}
                    disabled={!isRunning || pendingAction !== null}
                >
                    {pendingAction === 'stop' ? 'Stopping...' : 'Stop Evolution'}
                </button>
                <button
                    type="button"
                    className={`${styles.controlButton} ${styles.controlMuted}`}
                    onClick={handleSaveCheckpoint}
                    disabled={generationHistory.length === 0 || pendingAction !== null}
                >
                    {pendingAction === 'checkpoint' ? 'Saving...' : 'Save Checkpoint'}
                </button>
            </div>

            <div className={styles.overviewCards}>
                <Card label="Generations elapsed" value={String(overview.generationsElapsed)} />
                <Card label="Genomes evaluated" value={String(overview.genomesEvaluated)} />
                <Card
                    label="Current best fitness"
                    value={overview.currentBestFitness !== undefined ? overview.currentBestFitness.toFixed(4) : 'data unavailable'}
                />
                <Card label="Pareto front size" value={String(overview.paretoFrontSize)} />
                <Card label="Elapsed time" value={formatDuration(overview.elapsedTimeSeconds)} />
                <Card label="ETA" value={formatEta(overview.etaSeconds)} />
                <Card
                    label="Feasible ratio"
                    value={
                        overview.feasibleTotal > 0
                            ? `${((overview.feasibleSolutions / overview.feasibleTotal) * 100).toFixed(1)}%`
                            : 'data unavailable'
                    }
                />
            </div>

            <div className={styles.chartsAndJobs}>
                <div className={styles.chartPanel}>
                    <h3 className={styles.chartTitle}>Fitness and feasibility timeline</h3>
                    {fitnessTimeline.length > 0 ? (
                        <Line data={chartData} options={chartOptions} />
                    ) : (
                        <div className={styles.unavailable}>data unavailable</div>
                    )}
                </div>

                <div className={styles.jobsPanel}>
                    <div className={styles.jobsHeader}>
                        <h3 className={styles.chartTitle}>Active jobs</h3>
                        <div className={styles.jobsFilters}>
                            <button
                                type="button"
                                className={`${styles.filterButton} ${showRunningOnly ? styles.filterButtonActive : ''}`}
                                onClick={() => setShowRunningOnly((v) => !v)}
                            >
                                running
                            </button>
                            <button
                                type="button"
                                className={`${styles.filterButton} ${showFailedOnly ? styles.filterButtonActive : ''}`}
                                onClick={() => setShowFailedOnly((v) => !v)}
                            >
                                failed
                            </button>
                        </div>
                    </div>

                    <div className={styles.jobsTableWrap}>
                        <table className={styles.jobsTable}>
                            <thead>
                                <tr>
                                    <th>Job</th>
                                    <th>Genome</th>
                                    <th>Stage</th>
                                    <th>Progress</th>
                                    <th>Status</th>
                                    <th>Duration</th>
                                    <th>ETA</th>
                                </tr>
                            </thead>
                            <tbody>
                                {visibleJobs.map((job) => (
                                    <tr key={job.jobId}>
                                        <td>{job.jobId}</td>
                                        <td>{job.genomeId}</td>
                                        <td>{job.stage}</td>
                                        <td>{job.progressPercent}%</td>
                                        <td>
                                            <span className={`${styles.status} ${statusClass(job.status) ? styles[statusClass(job.status)!] : ''}`}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td>{(job.durationMs / 1000).toFixed(2)}s</td>
                                        <td>{formatEta(job.etaSeconds)}</td>
                                    </tr>
                                ))}
                                {visibleJobs.length === 0 && (
                                    <tr>
                                        <td colSpan={7} className={styles.unavailable}>data unavailable</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div className={styles.tabs}>
                <div className={styles.tabList}>
                    <button
                        type="button"
                        className={`${styles.tabButton} ${tab === 'pareto' ? styles.tabButtonActive : ''}`}
                        onClick={() => setTab('pareto')}
                    >
                        Pareto Front
                    </button>
                    <button
                        type="button"
                        className={`${styles.tabButton} ${tab === 'genealogy' ? styles.tabButtonActive : ''}`}
                        onClick={() => setTab('genealogy')}
                    >
                        Genealogy Tree
                    </button>
                    <button
                        type="button"
                        className={`${styles.tabButton} ${tab === 'stopping' ? styles.tabButtonActive : ''}`}
                        onClick={() => setTab('stopping')}
                    >
                        Stopping Criteria
                    </button>
                    <button
                        type="button"
                        className={`${styles.tabButton} ${tab === 'performance' ? styles.tabButtonActive : ''}`}
                        onClick={() => setTab('performance')}
                    >
                        Performance Metrics
                    </button>
                </div>

                <div className={styles.tabBody}>
                    {tab === 'pareto' && (
                        <ParetoFrontVisualizer
                            currentParetoFront={currentParetoFront}
                            paretoHistory={paretoHistory}
                            feasibilityByGenomeId={feasibilityByGenomeId}
                            constraintViolationScoreByGenomeId={constraintViolationScoreByGenomeId}
                            showOnlyFeasible={showOnlyFeasible}
                            onUseAsSeed={onUseAsSeed}
                            onOpenDetails={onOpenGenomeDetails}
                            onExportSelected={onExportSelected}
                        />
                    )}

                    {tab === 'genealogy' && (
                        <GenealogicTreeView
                            genealogyTree={genealogyTree}
                            paretoHistory={paretoHistory}
                            onOpenGenomeDetails={onOpenGenomeDetails}
                            onGenealogyTreeSync={onGenealogyTreeSync}
                        />
                    )}

                    {tab === 'stopping' && (
                        <>
                            {isRunning ? (
                                <StoppingCriteriaLiveMonitor
                                    isRunning={isRunning}
                                    generation={generation}
                                    elapsedSeconds={elapsedRuntimeSeconds}
                                    bestAccuracy={bestAccuracyNormalized}
                                />
                            ) : (
                                <StoppingCriteriaSummary
                                    triggeredCriterionIndex={triggeredCriterionIndex}
                                    criteria={stoppingCriteria as any}
                                    finalGeneration={generation}
                                    elapsedSeconds={elapsedRuntimeSeconds}
                                    finalAccuracy={bestAccuracyNormalized}
                                />
                            )}
                        </>
                    )}

                    {tab === 'performance' && (
                        <>
                            {currentSnapshot && currentSnapshot.genomes.length > 0 ? (
                                <>
                                    <table className={styles.performanceTable}>
                                        <thead>
                                            <tr>
                                                <th>Genome</th>
                                                <th>Train Duration</th>
                                                <th>Inference Latency</th>
                                                <th>Peak Active Memory</th>
                                                <th>Memory Breakdown</th>
                                                <th>Samples/sec</th>
                                                <th>Profiler</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {currentSnapshot.genomes.map((genome) => (
                                                <tr key={genome.id}>
                                                    <td>{genome.id}</td>
                                                    <td>{((genome.profiler?.total_train_duration_ms ?? 0) / 1000).toFixed(2)}s</td>
                                                    <td>{(genome.profiler?.inference_msec_per_sample ?? 0).toFixed(3)}ms</td>
                                                    <td>{formatMemory(genome.profiler?.peak_active_memory_mb)}</td>
                                                    <td>
                                                        model {formatMemory(genome.profiler?.peak_model_params_mb)} | grad {formatMemory(genome.profiler?.peak_gradient_mb)} | opt {formatMemory(genome.profiler?.peak_optim_state_mb)} | act {formatMemory(genome.profiler?.peak_activation_mb)}
                                                    </td>
                                                    <td>{(genome.profiler?.samples_per_sec ?? 0).toFixed(1)}</td>
                                                    <td>
                                                        <button
                                                            type="button"
                                                            className={styles.inlineAction}
                                                            onClick={() => onOpenProfiler(genome)}
                                                            disabled={!genome.profiler}
                                                        >
                                                            Open
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>

                                    <GenerationStatsTable
                                        generations={generationHistory}
                                        selectedGeneration={currentSnapshot.generation}
                                    />
                                </>
                            ) : (
                                <div className={styles.unavailable}>data unavailable</div>
                            )}
                        </>
                    )}
                </div>
            </div>

            <div className={styles.eventLog}>
                <h4 className={styles.logTitle}>Event log</h4>
                <div className={styles.logList}>
                    {logs.length === 0 && <div className={`${styles.logEntry} ${styles.logMuted}`}>data unavailable</div>}
                    {logs.map((log, index) => (
                        <div key={`${log.time}-${index}`} className={styles.logEntry}>
                            [{log.time}] {log.message}
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}

function statusClass(status: 'running' | 'completed' | 'failed' | 'queued'): string | null {
    if (status === 'running') return 'statusRunning';
    if (status === 'completed') return 'statusCompleted';
    if (status === 'failed') return 'statusFailed';
    if (status === 'queued') return 'statusQueued';
    return null;
}

function Card({ label, value }: { label: string; value: string }) {
    return (
        <article className={styles.card}>
            <div className={styles.cardLabel}>{label}</div>
            <div className={styles.cardValue}>{value}</div>
        </article>
    );
}
