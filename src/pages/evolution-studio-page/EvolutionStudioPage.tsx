import React, { useState, useEffect, useMemo } from 'react';
import { TitleBar } from '../../widgets/title-bar/TitleBar';
import styles from './EvolutionStudioPage.module.css';
import { BsPlay, BsPlus, BsX } from 'react-icons/bs';
import { useEvolutionLoop } from '../../features/evolution-studio';
import {
    evaluateGenomeFeasibility,
    useEvolutionSettingsStore,
    StoppingCriteriaLiveMonitor,
    StoppingCriteriaSummary,
    type DeviceConstraintParams,
} from '../../features/evolution-manager';
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
import { ExportGenomeWithWeightsModal } from '../../features/evolution-studio/ui/ExportGenomeWithWeightsModal';
import { EvolutionDashboard } from '../../widgets/evolution-dashboard';
import { PostEvolutionPanel } from '../../widgets/post-evolution-panel';
import type { GenerationParetoFront, GenomeObjectives } from '../../shared/lib';

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
    const datasetProfileId = useDatasetManagerStore(state => state.selectedProfileId);
    const profiles = useDatasetManagerStore(state => state.profiles);
    const settings = useEvolutionSettingsStore();
    const setParetoForGeneration = useEvolutionSettingsStore(
        (state) => state.setParetoForGeneration,
    );
    const setGenerationProfilingStat = useEvolutionSettingsStore(
        (state) => state.setGenerationProfilingStat,
    );
    const setGenealogyTree = useEvolutionSettingsStore(
        (state) => state.setGenealogyTree,
    );
    const {
        isRunning,
        isPaused,
        startEvolution,
        stopEvolution,
        pauseEvolution,
        resumeEvolution,
        saveCheckpoint,
        generation,
        population,
        hallOfFame,
        logs,
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
    const [exportGenomeId, setExportGenomeId] = useState<string | null>(null);
    const [stoppingTriggeredIndex, setStoppingTriggeredIndex] = useState<number | null>(null);
    const [evolutionCompleted, setEvolutionCompleted] = useState(false);
    const [elapsedRuntimeSeconds, setElapsedRuntimeSeconds] = useState(0);

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

        setGenerationProfilingStat(evaluated.generation, {
            generation: evaluated.generation,
            totalTrainingMs: evaluated.totalTrainingMs ?? 0,
            totalInferenceMs: evaluated.totalInferenceMs ?? 0,
            avgSamplesPerSec: evaluated.avgSamplesPerSec ?? 0,
            peakConcurrentVramMb,
            totalJobsCompleted: evaluated.genomes.length,
            totalJobsFailed: 0,
        });

        if (evaluated.genealogy && evaluated.genealogy.size > 0) {
            setGenealogyTree(new Map(evaluated.genealogy));
        }
    }, [generationHistory, setGenerationProfilingStat, setGenealogyTree]);

    // Track evolution completion
    useEffect(() => {
        if (!isRunning && generationHistory.length > 0) {
            setEvolutionCompleted(true);
        }
    }, [isRunning, generationHistory.length]);

    useEffect(() => {
        if (!isRunning) {
            return;
        }

        const startMs = Date.now();
        setElapsedRuntimeSeconds(0);

        const intervalId = setInterval(() => {
            setElapsedRuntimeSeconds(Math.floor((Date.now() - startMs) / 1000));
        }, 1000);

        return () => clearInterval(intervalId);
    }, [isRunning]);

    // Current snapshot (latest generation)
    const currentSnapshot = generationHistory.length > 0 
        ? generationHistory[generationHistory.length - 1] 
        : undefined;
    const profilerStats = useProfilerStats(currentSnapshot?.genomes ?? []);
    const bestAccuracyNormalized = useMemo(() => {
        const best = population.reduce((max, genome) => Math.max(max, genome.accuracy ?? 0), 0);
        return best > 1 ? best / 100 : best;
    }, [population]);

    useEffect(() => {
        if (isRunning || !evolutionCompleted) {
            return;
        }

        const criteria = settings.stoppingPolicy.criteria;
        const triggeredFromBackend = settings.currentStoppingProgress.triggeredCriteria ?? [];

        if (triggeredFromBackend.length > 0) {
            const byTypeIndex = criteria.findIndex((criterion) =>
                triggeredFromBackend.some((triggered) => triggered.includes(criterion.type)),
            );
            setStoppingTriggeredIndex(byTypeIndex >= 0 ? byTypeIndex : null);
            return;
        }

        const heuristicIndex = criteria.findIndex((criterion) => {
            if (criterion.type === 'GenerationLimit') {
                return generation >= criterion.max_generations;
            }
            if (criterion.type === 'TimeLimit') {
                return elapsedRuntimeSeconds >= criterion.max_seconds;
            }
            if (criterion.type === 'TargetAccuracy') {
                return bestAccuracyNormalized >= criterion.threshold;
            }
            return false;
        });

        setStoppingTriggeredIndex(heuristicIndex >= 0 ? heuristicIndex : null);
    }, [
        isRunning,
        evolutionCompleted,
        settings.stoppingPolicy.criteria,
        settings.currentStoppingProgress.triggeredCriteria,
        generation,
        elapsedRuntimeSeconds,
        bestAccuracyNormalized,
    ]);

    const activeProfile = profiles.find(p => p.id === datasetProfileId);
    const activeDeviceConstraints = useMemo<DeviceConstraintParams | undefined>(() => {
        if (settings.customDeviceParams) {
            const fromCustom = {
                mops_budget:
                    settings.customDeviceParams.mops_budget ??
                    Math.max(1, settings.resourceTargets.macs / 1_000_000),
                ram_mb: settings.customDeviceParams.ram_mb,
                flash_mb:
                    settings.customDeviceParams.flash_mb ??
                    settings.customDeviceParams.max_model_size_mb ??
                    Math.max(1, settings.resourceTargets.flash / (1024 * 1024)),
                latency_budget_ms: settings.customDeviceParams.latency_budget_ms,
            };

            if (
                fromCustom.mops_budget <= 0 ||
                fromCustom.ram_mb <= 0 ||
                fromCustom.flash_mb <= 0 ||
                fromCustom.latency_budget_ms <= 0
            ) {
                return undefined;
            }

            return fromCustom;
        }

        if (settings.selectedDeviceProfile) {
            const fallbackFlash =
                settings.selectedDeviceProfile.max_model_size_mb ??
                Math.max(1, settings.resourceTargets.flash / (1024 * 1024));
            return {
                mops_budget: Math.max(1, settings.resourceTargets.macs / 1_000_000),
                ram_mb: settings.selectedDeviceProfile.ram_mb,
                flash_mb: fallbackFlash,
                latency_budget_ms: settings.selectedDeviceProfile.inference_latency_budget_ms,
            };
        }

        return undefined;
    }, [
        settings.customDeviceParams,
        settings.resourceTargets.flash,
        settings.resourceTargets.macs,
        settings.selectedDeviceProfile,
    ]);

    const feasibilityByGenomeId = useMemo<Record<string, boolean>>(() => {
        const result: Record<string, boolean> = {};
        if (!activeDeviceConstraints) {
            return result;
        }

        for (const [_, front] of settings.paretoHistory) {
            const all = front.all_genomes ?? front.pareto_members;
            for (const objective of all) {
                result[objective.genome_id] = evaluateGenomeFeasibility(
                    objective,
                    activeDeviceConstraints,
                ).isFeasible;
            }
        }

        return result;
    }, [activeDeviceConstraints, settings.paretoHistory]);

    const constraintViolationScoreByGenomeId = useMemo<Record<string, number>>(() => {
        const result: Record<string, number> = {};
        if (!activeDeviceConstraints) {
            return result;
        }

        for (const [_, front] of settings.paretoHistory) {
            const all = front.all_genomes ?? front.pareto_members;
            for (const objective of all) {
                result[objective.genome_id] = evaluateGenomeFeasibility(
                    objective,
                    activeDeviceConstraints,
                ).violationScore;
            }
        }

        return result;
    }, [activeDeviceConstraints, settings.paretoHistory]);

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

        setParetoForGeneration(currentSnapshot.generation, payload);
    }, [currentSnapshot, setParetoForGeneration]);

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
        setExportGenomeId(genomeId);
    };

    const handleStart = async () => {
        try {
            // Reset stopping criteria tracking
            setStoppingTriggeredIndex(null);
            setEvolutionCompleted(false);

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

                            {isRunning && (
                                <div className={styles.setupCard}>
                                    <StoppingCriteriaLiveMonitor
                                        isRunning={isRunning}
                                        generation={generation}
                                        elapsedSeconds={elapsedRuntimeSeconds}
                                        bestAccuracy={bestAccuracyNormalized}
                                    />
                                </div>
                            )}

                            {!isRunning && evolutionCompleted && generationHistory.length > 0 && (
                                <div className={styles.setupCard}>
                                    <StoppingCriteriaSummary
                                        triggeredCriterionIndex={stoppingTriggeredIndex}
                                        criteria={settings.stoppingPolicy.criteria}
                                        finalGeneration={generation}
                                        elapsedSeconds={elapsedRuntimeSeconds}
                                        finalAccuracy={bestAccuracyNormalized}
                                    />
                                </div>
                            )}
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

                    <EvolutionDashboard
                        isRunning={isRunning}
                        isPaused={isPaused}
                        generation={generation}
                        generationHistory={generationHistory}
                        liveMetrics={liveMetrics}
                        currentEvaluatingIndex={currentEvaluatingIndex}
                        population={population}
                        logs={logs}
                        elapsedRuntimeSeconds={elapsedRuntimeSeconds}
                        useMaxGenerations={settings.useMaxGenerations}
                        maxGenerations={settings.maxGenerations}
                        currentParetoFront={settings.currentParetoFront}
                        paretoHistory={settings.paretoHistory}
                        feasibilityByGenomeId={feasibilityByGenomeId}
                        constraintViolationScoreByGenomeId={constraintViolationScoreByGenomeId}
                        showOnlyFeasible={settings.showOnlyFeasible}
                        genealogyTree={settings.genealogyTree}
                        onGenealogyTreeSync={settings.setGenealogyTree}
                        onUseAsSeed={handleUseParetoAsSeed}
                        onOpenGenomeDetails={handleOpenParetoDetails}
                        onExportSelected={handleExportParetoSelected}
                        stoppingCriteria={settings.stoppingPolicy.criteria}
                        triggeredCriterionIndex={stoppingTriggeredIndex}
                        bestAccuracyNormalized={bestAccuracyNormalized}
                        onPause={pauseEvolution}
                        onResume={resumeEvolution}
                        onStop={stopEvolution}
                        onSaveCheckpoint={saveCheckpoint}
                        onOpenProfiler={(genome) => setProfilerGenome(genome)}
                    />

                    {!isRunning && evolutionCompleted && generationHistory.length > 0 && (
                        <div className={styles.postRunPanelWrap}>
                            <PostEvolutionPanel
                                paretoHistory={settings.paretoHistory}
                                genealogyTree={settings.genealogyTree}
                                onSyncGenealogyTree={settings.setGenealogyTree}
                                onOpenGenomeDetails={handleOpenParetoDetails}
                                onExportWeights={handleExportParetoSelected}
                                onContinueEvolution={handleStart}
                                generation={generation}
                                elapsedRuntimeSeconds={elapsedRuntimeSeconds}
                                stoppingPolicy={settings.stoppingPolicy.criteria}
                                stoppingReason={
                                    stoppingTriggeredIndex !== null
                                        ? `Triggered: ${settings.stoppingPolicy.criteria[stoppingTriggeredIndex]?.type}`
                                        : 'Completed'
                                }
                                genomeById={genomeById}
                                activeDeviceConstraints={activeDeviceConstraints}
                                feasibilityByGenomeId={feasibilityByGenomeId}
                                constraintViolationScoreByGenomeId={constraintViolationScoreByGenomeId}
                            />
                        </div>
                    )}

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

            {exportGenomeId && (
                <ExportGenomeWithWeightsModal
                    genomeId={exportGenomeId}
                    onClose={() => setExportGenomeId(null)}
                />
            )}
        </div>
    );
};
