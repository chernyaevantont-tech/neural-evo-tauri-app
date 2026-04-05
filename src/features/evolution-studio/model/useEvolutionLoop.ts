import { useState, useCallback, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useEvolutionSettingsStore, getAdaptiveMutationRates } from '../../evolution-manager/model/store';
import { useDatasetManagerStore } from '../../../features/dataset-manager/model/store';
import { Genome, BaseNode, serializeGenome, deserializeGenome, generateRandomArchitecture, extractShapesFromDatasetProfile } from '../../../entities/canvas-genome';
import { computeZeroCostScore, ZeroCostMetrics } from './useZeroCostEvaluation';
import { createObjectiveVector, checkConstraints, ObjectiveVector } from './multiObjectiveFitness';
import { nonDominatedSorting } from './paretoSorting';
import { ConvergenceChecker } from './convergenceChecker';

export interface EvaluationResult {
    genome_id: string;
    loss: number;
    accuracy: number;
}

export interface PopulatedGenome {
    id: string;
    genome: Genome;
    nodes: BaseNode[];
    loss?: number;
    accuracy?: number;
    adjustedFitness?: number;
    trainingMetrics?: BatchMetrics[];
    resources?: { totalFlash: number; totalRam: number; totalMacs: number; totalNodes: number };
    zeroCostMetric?: ZeroCostMetrics;
    // Multi-objective fields
    objectives?: ObjectiveVector;
    paretoRank?: number;
    crowdingDistance?: number;
    isFeasible?: boolean;
}

export interface LogEntry {
    time: string;
    message: string;
    type: 'info' | 'success' | 'warn' | 'error';
}

export interface GenerationStat {
    generation: number;
    bestFitness: number;
    avgNodes: number;
}

export interface BatchMetrics {
    epoch: number;
    batch: number;
    total_batches: number;
    loss: number;
    accuracy: number;
}

export interface GenomeResultEvent {
    index: number;
    loss: number;
    accuracy: number;
}

export interface GenerationSnapshot {
    generation: number;
    genomes: PopulatedGenome[];
    bestFitness: number;
    avgNodes: number;
    timestamp: string;
    evaluated: boolean;  // false = pre-eval, true = post-eval with fitness
}

export const useEvolutionLoop = (datasetProfileId: string | null) => {
    const settings = useEvolutionSettingsStore();

    const [isRunning, setIsRunning] = useState(false);
    const [generation, setGeneration] = useState(0);
    const [population, setPopulation] = useState<PopulatedGenome[]>([]);
    const [hallOfFame, setHallOfFame] = useState<PopulatedGenome[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [stats, setStats] = useState<GenerationStat[]>([]);

    const [currentEvaluatingIndex, setCurrentEvaluatingIndex] = useState<number>(0);
    const [liveMetrics, setLiveMetrics] = useState<BatchMetrics[]>([]);
    const [generationHistory, setGenerationHistory] = useState<GenerationSnapshot[]>([]);
    
    // Pareto Archive for multi-objective mode
    const [paretoArchive, setParetoArchive] = useState<PopulatedGenome[]>([]);

    // Per-genome metrics accumulator (ref to avoid re-renders on every batch)
    const perGenomeMetricsRef = useRef<Map<number, BatchMetrics[]>>(new Map());
    const activeGenomeIndexRef = useRef<number>(0);
    
    // Convergence checker for multi-objective mode
    const convergenceCheckerRef = useRef<ConvergenceChecker | null>(null);

    // Using refs for safe async access within loops
    const isRunningRef = useRef(false);

    useEffect(() => {
        let unlistenGenome: (() => void) | undefined;
        let unlistenStart: (() => void) | undefined;
        let unlistenBatch: (() => void) | undefined;
        let unlistenResult: (() => void) | undefined;

        import('@tauri-apps/api/event').then(({ listen }) => {
            listen<number>('evaluating-genome', (event) => {
                setCurrentEvaluatingIndex(event.payload);
            }).then(fn => {
                unlistenGenome = fn;
            });

            listen<number>('evaluating-genome-start', (event) => {
                setLiveMetrics([]); // Clear live charts for the new genome
                activeGenomeIndexRef.current = event.payload;
                perGenomeMetricsRef.current.set(event.payload, []);
                // Add log for the user
                addLog(`Starting evaluation for Genome #${event.payload + 1}...`, 'info');
            }).then(fn => {
                unlistenStart = fn;
            });

            listen<BatchMetrics>('evaluating-batch-metrics', (event) => {
                setLiveMetrics(prev => [...prev, event.payload]);
                // Also store per-genome
                const idx = activeGenomeIndexRef.current;
                const arr = perGenomeMetricsRef.current.get(idx);
                if (arr) arr.push(event.payload);
            }).then(fn => {
                unlistenBatch = fn;
            });

            // Progressive per-genome result update
            listen<GenomeResultEvent>('evaluating-genome-result', (event) => {
                const { index, loss, accuracy } = event.payload;
                const genomeMetrics = perGenomeMetricsRef.current.get(index) || [];
                setGenerationHistory(prev => {
                    if (prev.length === 0) return prev;
                    const updated = [...prev];
                    const last = { ...updated[updated.length - 1] };
                    const genomes = [...last.genomes];
                    if (index < genomes.length) {
                        genomes[index] = {
                            ...genomes[index],
                            loss,
                            accuracy,
                            trainingMetrics: [...genomeMetrics],
                            resources: genomes[index].genome.GetGenomeResources()
                        };
                    }
                    last.genomes = genomes;
                    updated[updated.length - 1] = last;
                    return updated;
                });
            }).then(fn => {
                unlistenResult = fn;
            });
        });

        return () => {
            if (unlistenGenome) unlistenGenome();
            if (unlistenStart) unlistenStart();
            if (unlistenBatch) unlistenBatch();
            if (unlistenResult) unlistenResult();
        };
    }, []);

    const addLog = useCallback((msg: string, type: LogEntry['type'] = 'info') => {
        setLogs(prev => [...prev.slice(-49), { time: new Date().toLocaleTimeString(), message: msg, type }]);
    }, []);

    const stopEvolution = useCallback(() => {
        setIsRunning(false);
        isRunningRef.current = false;
        // Signal the Rust backend to stop the current evaluation pass
        invoke('stop_evolution').catch(err => console.error('Failed to stop backend:', err));
        addLog("Evolution stopped by user.", "warn");
    }, [addLog]);

    // Initial Spawning (from multiple library seeds or a fallback graph)
    const initPopulation = useCallback(async (seedJSONs: string[]) => {
        const popSize = settings.populationSize;
        const genomes: PopulatedGenome[] = [];
        try {
            // Determine how many genomes to generate randomly vs from seeds
            // Case 1: useRandomInitialization enabled with seeds → split by ratio
            // Case 2: useRandomInitialization enabled without seeds → all random
            // Case 3: useRandomInitialization disabled → all from seeds
            let numRandom = 0;
            let numFromSeeds = popSize;
            
            if (settings.useRandomInitialization) {
                if (seedJSONs.length > 0) {
                    // Split between random and seeded
                    numRandom = Math.floor((popSize * settings.randomInitRatio) / 100);
                    numFromSeeds = popSize - numRandom;
                } else {
                    // Pure random initialization (no seeds available)
                    numRandom = popSize;
                    numFromSeeds = 0;
                }
            }

            // Get dataset profile for random generation
            let inputShape: number[] | null = null;
            let outputShape: number[] | null = null;
            let dataTypeHint: 'Image' | 'TemporalSequence' | 'Vector' | undefined = undefined;

            if (numRandom > 0) {
                const profiles = useDatasetManagerStore.getState().profiles;
                const profile = profiles.find(p => p.id === datasetProfileId);

                if (profile && profile.streams && profile.streams.length > 0) {
                    const shapes = extractShapesFromDatasetProfile(profile.streams);
                    if (shapes) {
                        inputShape = shapes.inputShape;
                        outputShape = shapes.outputShape;
                        dataTypeHint = shapes.dataTypeHint;
                        
                        const typeHintStr = dataTypeHint ? ` (${dataTypeHint})` : '';
                        addLog(`Detected input shape: [${inputShape.join(',')}]${typeHintStr}, output shape: [${outputShape.join(',')}]`, "info");
                    }
                }
            }

            // First pass: generate random architectures if enabled
            if (numRandom > 0 && inputShape && outputShape) {
                let randomAttempts = 0;
                while (genomes.length < numRandom && randomAttempts < numRandom * 5) {
                    randomAttempts++;
                    try {
                        const randomGenome = generateRandomArchitecture(inputShape, outputShape, {
                            maxDepth: 8,
                            useAttention: false,
                            dataTypeHint: dataTypeHint
                        });
                        const nodes = randomGenome.getAllNodes();

                        if (!Genome.isGenomeFeasible(nodes)) {
                            addLog(`isGenomeFeasible returned false.`, "warn");
                            continue;
                        }

                        // Apply parameter mutations for diversity
                        const mutationOpts = new Map<string, number>();
                        mutationOpts.set('params', settings.mutationRates.params || 0.5);
                        randomGenome.getAllNodes().forEach(node => {
                            if (typeof (node as any).Mutate === 'function') {
                                (node as any).Mutate(mutationOpts);
                            }
                        });

                        genomes.push({
                            id: crypto.randomUUID(),
                            genome: randomGenome,
                            nodes: nodes
                        });
                    } catch (e) {
                        console.warn(`[initPopulation] Random generation failed. Attempt ${randomAttempts}/${numRandom * 5}`, e);
                        addLog(`Exception: ${e}`, "error");
                        continue;
                    }
                }

                if (genomes.length > 0) {
                    addLog(`Generated ${genomes.length} random architectures from dataset shapes.`, "info");
                }
            }

            // Second pass: instantiate seed genomes
            const seedInstances = [];
            for (const seedStr of seedJSONs) {
                const { genome, nodes } = await deserializeGenome(seedStr);

                if (!Genome.isGenomeFeasible(nodes)) {
                    addLog(`Seed genome is architecturally invalid or excessive. Skipping.`, "warn");
                    continue;
                }

                seedInstances.push(genome);
                genomes.push({
                    id: crypto.randomUUID(),
                    nodes: nodes,
                    genome: genome
                });
            }

            // Final fallback if no genomes at all
            if (genomes.length === 0 && seedInstances.length === 0) {
                addLog("No seeds or random genomes available, initialization aborted.", "error");
                return;
            }

            // If no seeds but we have random genomes, use them for mutation base
            if (seedInstances.length === 0 && genomes.length > 0) {
                for (const g of genomes) {
                    seedInstances.push(g.genome);
                }
            }

            // Fill the rest of the population up to popSize by mutating from available sources
            let initAttempts = 0;
            while (genomes.length < popSize && initAttempts < popSize * 10) {
                initAttempts++;
                try {
                    // Pick a random seed or random genome to mutate
                    const baseSeed = seedInstances[Math.floor(Math.random() * seedInstances.length)];

                    // Clone the seed
                    const seedStr = await serializeGenome(baseSeed);
                    let { genome: clone } = await deserializeGenome(seedStr);

                    // === Guaranteed diversity: force at least one structural mutation ===
                    const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;
                    let mutated = false;
                    const maxMutAttempts = 10;

                    for (let attempt = 0; attempt < maxMutAttempts && !mutated; attempt++) {
                        // Pick a random structural mutation and force-apply it
                        const structuralMutations = [
                            () => clone.MutateAddNode(maxNodes),
                            () => clone.MutateRemoveNode(),
                            () => clone.MutateRemoveSubgraph(),
                            ...(settings.mutationRates.addSkipConnection ? [() => clone.MutateAddSkipConnection(maxNodes)] : []),
                            ...(settings.mutationRates.changeLayerType ? [() => clone.MutateChangeLayerType(maxNodes)] : []),
                        ];

                        const pick = structuralMutations[Math.floor(Math.random() * structuralMutations.length)];
                        const res = pick();
                        if (res) {
                            clone = res.genome;
                            mutated = true;
                        }
                    }

                    // Additional probabilistic mutation rounds for extra variance
                    const extraRounds = Math.floor(Math.random() * 3);
                    for (let r = 0; r < extraRounds; r++) {
                        const dynamicRates = settings.useAdaptiveMutation
                            ? getAdaptiveMutationRates(clone.getAllNodes().length)
                            : {
                                addNode: settings.mutationRates.addNode,
                                removeNode: settings.mutationRates.removeNode,
                                removeSubgraph: settings.mutationRates.removeSubgraph
                            };

                        if (Math.random() < dynamicRates.removeSubgraph) {
                            const res = clone.MutateRemoveSubgraph();
                            if (res) clone = res.genome;
                        }
                        if (Math.random() < dynamicRates.removeNode) {
                            const res = clone.MutateRemoveNode();
                            if (res) clone = res.genome;
                        }
                        if (Math.random() < dynamicRates.addNode) {
                            const res = clone.MutateAddNode(maxNodes);
                            if (res) clone = res.genome;
                        }
                        if (settings.mutationRates.addSkipConnection && Math.random() < settings.mutationRates.addSkipConnection) {
                            const res = clone.MutateAddSkipConnection(maxNodes);
                            if (res) clone = res.genome;
                        }
                        if (settings.mutationRates.changeLayerType && Math.random() < settings.mutationRates.changeLayerType) {
                            const res = clone.MutateChangeLayerType(maxNodes);
                            if (res) clone = res.genome;
                        }
                    }

                    // Always apply params mutation to ensure weight diversity
                    {
                        const mutationOpts = new Map<string, number>();
                        mutationOpts.set('params', settings.mutationRates.params || 0.5);
                        clone.getAllNodes().forEach(node => {
                            if (typeof (node as any).Mutate === 'function') {
                                (node as any).Mutate(mutationOpts);
                            }
                        });
                    }

                    const finalNodes = clone.getAllNodes();
                    if (!Genome.isGenomeFeasible(finalNodes)) {
                        console.warn(`[initPopulation] Mutated clone failed feasibility check. Retrying.`);
                        if (initAttempts % 10 === 0) {
                            addLog(`Retrying mutation (structural mismatch or budget limit hit)...`, "warn");
                        }
                        continue;
                    }

                    genomes.push({
                        id: crypto.randomUUID(),
                        genome: clone,
                        nodes: finalNodes
                    });
                } catch (e) {
                    console.warn(`[initPopulation] Seed mutation failed (likely shape mismatch). Attempt ${initAttempts}/${popSize * 10}`, e);
                    continue;
                }
            }

            // If we have more seeds than popSize, truncate
            if (genomes.length > popSize) {
                genomes.length = popSize;
            }

            setPopulation(genomes);
            setGeneration(0);
            setStats([]);

            // Save pre-eval snapshot so genomes are visible immediately
            setGenerationHistory([{
                generation: 0,
                genomes: [...genomes],
                bestFitness: 0,
                avgNodes: Math.round(genomes.reduce((acc, g) => acc + g.nodes.length, 0) / genomes.length),
                timestamp: new Date().toLocaleTimeString(),
                evaluated: false
            }]);

            addLog(`Spawned Generation 0: ${numFromSeeds} direct seeds, ${numRandom} random archs, ${Math.max(0, genomes.length - numFromSeeds - numRandom)} mutated clones.`, "info");
        } catch (e) {
            addLog(`Failed to initialize population: ${String(e)}`, "error");
        }
    }, [addLog, settings, datasetProfileId]);

    // Main Async Loop
    const runGeneration = useCallback(async () => {
        if (!isRunningRef.current || population.length === 0 || !datasetProfileId) return;

        addLog(`--- Starting Generation ${generation} Evaluation ---`, "info");

        try {
            // 1. Serialize population for Rust
            const serializedGenomes = await Promise.all(
                population.map(p => serializeGenome(p.genome))
            );

            addLog(`Sending ${serializedGenomes.length} genomes to Rust for evaluation...`);

            // 1b. Zero-Cost Proxy Scoring (if enabled)
            let zeroCostScores: ZeroCostMetrics[] = [];
            let evalEpochsAdjustments: Map<number, number> = new Map();
            
            if (settings.useZeroCostProxies) {
                addLog(`Computing zero-cost proxy scores for architecture evaluation...`);
                
                for (let i = 0; i < serializedGenomes.length; i++) {
                    try {
                        const score = await computeZeroCostScore(serializedGenomes[i], {
                            enabled: true,
                            strategy: settings.zeroCostStrategy,
                            fastPassThreshold: settings.fastPassThreshold,
                            partialTrainingEpochs: settings.partialTrainingEpochs,
                            useVoting: false,
                        });
                        
                        zeroCostScores.push(score);
                        
                        // Determine epochs for this genome based on strategy decision
                        const recommended = score.strategy_decision === 'skip' 
                            ? 0 
                            : score.strategy_decision === 'partial_train'
                                ? settings.partialTrainingEpochs
                                : settings.evalEpochs;
                        
                        evalEpochsAdjustments.set(i, recommended);
                        
                        addLog(
                            `Genome ${i}: SynFlow=${score.synflow.toFixed(2)} ` +
                            `(${(score.normalized_score * 100).toFixed(0)}%) → ` +
                            `${score.strategy_decision === 'skip' ? 'SKIP' : recommended + ' epochs'}`,
                            "info"
                        );
                    } catch (e) {
                        // Fallback: full training if zero-cost fails
                        zeroCostScores.push({
                            synflow: 5.0,
                            normalized_score: 0.5,
                            strategy_decision: 'full_train',
                        });
                        evalEpochsAdjustments.set(i, settings.evalEpochs);
                        addLog(`Genome ${i}: Zero-cost scoring failed, using full training fallback`);
                    }
                }
                
                // Count statistics
                const skipped = zeroCostScores.filter(s => s.strategy_decision === 'skip').length;
                const partial = zeroCostScores.filter(s => s.strategy_decision === 'partial_train').length;
                const full = zeroCostScores.filter(s => s.strategy_decision === 'full_train').length;
                const avgSynFlow = zeroCostScores.reduce((sum, s) => sum + s.synflow, 0) / zeroCostScores.length;
                
                addLog(
                    `Zero-Cost Summary: ${full} full + ${partial} partial + ${skipped} skipped | ` +
                    `Avg SynFlow: ${avgSynFlow.toFixed(2)}`,
                    "success"
                );
            } else {
                // If zero-cost disabled, use standard epochs for all
                for (let i = 0; i < serializedGenomes.length; i++) {
                    evalEpochsAdjustments.set(i, settings.evalEpochs);
                }
            }

            // Look up dataset split percentages from the dataset manager
            const profiles = useDatasetManagerStore.getState().profiles;
            const currentProfile = profiles.find(p => p.id === datasetProfileId);
            const trainSplit = currentProfile?.split?.train ?? 80;
            const valSplit = currentProfile?.split?.val ?? 10;
            const testSplit = currentProfile?.split?.test ?? 10;

            // 2. Call Rust Evaluator
            // For now, we use a global evalEpochs value
            // In a more advanced implementation, we'd modify evaluate_population to accept per-genome epochs
            const perGenomeEpochs = Array.from({ length: serializedGenomes.length }, (_, i) => 
                evalEpochsAdjustments.get(i) ?? settings.evalEpochs
            );
            
            const results = await invoke<EvaluationResult[]>('evaluate_population', {
                genomes: serializedGenomes,
                datasetProfile: datasetProfileId,
                batchSize: settings.batchSize || 32,
                perGenomeEpochs,
                datasetPercent: settings.datasetPercent || 100,
                trainSplit,
                valSplit,
                testSplit,
            });

            // 3. Map Results & Apply Fitness (Parsimony + Resource-Aware + Zero-Cost)
            const alpha = settings.useParsimonyPressure ? settings.parsimonyAlpha : 0;
            const evaluatedPop = population.map((p, index) => {
                const res = results[index];
                const nodeCount = p.genome.getAllNodes().length;
                const resources = p.genome.GetGenomeResources();
                const zeroCostMetric = zeroCostScores.length > index ? zeroCostScores[index] : undefined;

                let baseFitness = res.accuracy > 0 ? res.accuracy : (1 / (1 + res.loss));

                // Combine with zero-cost proxy if available
                if (settings.useZeroCostProxies && zeroCostMetric) {
                    if (zeroCostMetric.strategy_decision === 'skip') {
                        // Genome was not trained, use only proxy score
                        baseFitness = zeroCostMetric.normalized_score;
                    } else {
                        // Genome was trained (partial or full), combine accuracy with proxy
                        // 70% accuracy + 30% proxy for diversity
                        baseFitness = (0.7 * baseFitness) + (0.3 * zeroCostMetric.normalized_score);
                    }
                }

                // === MULTI-OBJECTIVE MODE ===
                if (settings.useMultiObjective) {
                    // Create objective vector
                    const wasSkipped = zeroCostMetric?.strategy_decision === 'skip';
                    const objectives = createObjectiveVector(
                        res.accuracy,
                        zeroCostMetric?.normalized_score,
                        resources,
                        wasSkipped || false
                    );

                    // Check resource constraints
                    const constraintCheck = checkConstraints(objectives, {
                        maxFlashKB: settings.resourceConstraints.useHardConstraints 
                            ? settings.resourceConstraints.maxFlashKB 
                            : undefined,
                        maxRamKB: settings.resourceConstraints.useHardConstraints 
                            ? settings.resourceConstraints.maxRamKB 
                            : undefined,
                        maxMacs: settings.resourceConstraints.useHardConstraints 
                            ? settings.resourceConstraints.maxMacs 
                            : undefined
                    });
                    
                    // Attach per-genome training metrics
                    const genomeMetrics = perGenomeMetricsRef.current.get(index) || [];

                    return {
                        ...p,
                        loss: res.loss,
                        accuracy: res.accuracy,
                        objectives,
                        paretoRank: 0,  // Will be calculated after sorting
                        crowdingDistance: 0,
                        isFeasible: constraintCheck.isFeasible,
                        adjustedFitness: baseFitness,  // Keep for backward compat
                        trainingMetrics: genomeMetrics,
                        resources,
                        zeroCostMetric,
                    } as PopulatedGenome;
                }

                // === SINGLE-OBJECTIVE MODE (existing logic) ===
                
                // Resource-Aware Fitness penalty
                if (settings.useResourceAwareFitness) {
                    const flashPenalty = Math.max(0, resources.totalFlash - settings.resourceTargets.flash) / settings.resourceTargets.flash;
                    const ramPenalty = Math.max(0, resources.totalRam - settings.resourceTargets.ram) / settings.resourceTargets.ram;
                    const macsPenalty = Math.max(0, resources.totalMacs - settings.resourceTargets.macs) / settings.resourceTargets.macs;
                    const resourcePenalty = (flashPenalty + ramPenalty + macsPenalty) / 3;
                    baseFitness *= Math.max(0.1, 1.0 - resourcePenalty);
                }

                // Attach per-genome training metrics
                const genomeMetrics = perGenomeMetricsRef.current.get(index) || [];

                return {
                    ...p,
                    loss: res.loss,
                    accuracy: res.accuracy,
                    adjustedFitness: baseFitness - (alpha * nodeCount),
                    trainingMetrics: genomeMetrics,
                    resources,
                    zeroCostMetric,
                } as PopulatedGenome;
            });

            // === SORTING ===
            if (settings.useMultiObjective) {
                // Multi-objective: Pareto sorting
                const paretoItems = evaluatedPop.map(p => ({ id: p.id, vector: p.objectives! }));
                const rankings = nonDominatedSorting(paretoItems);
                
                // Apply rankings to population
                for (const p of evaluatedPop) {
                    const ranking = rankings.get(p.id)!;
                    p.paretoRank = ranking.rank;
                    p.crowdingDistance = ranking.crowdingDistance;
                }
                
                // Sort by Pareto rank then crowding distance
                evaluatedPop.sort((a, b) => {
                    const rankDiff = (a.paretoRank || 0) - (b.paretoRank || 0);
                    if (rankDiff !== 0) return rankDiff;
                    return (b.crowdingDistance || 0) - (a.crowdingDistance || 0);
                });
                
                const frontSize = evaluatedPop.filter(p => p.paretoRank === 0).length;
                addLog(`Generation ${generation} complete. Pareto front: ${frontSize} solutions.`, "success");
            } else {
                // Single-objective: Sort by adjusted fitness (descending)
                evaluatedPop.sort((a, b) => (b.adjustedFitness || 0) - (a.adjustedFitness || 0));
            }
            
            // Get best genome for stats (single-objective mode)
            const best = evaluatedPop[0];
            
            if (!settings.useMultiObjective) {
                addLog(`Generation ${generation} complete. Best Fitness: ${best.adjustedFitness?.toFixed(4)} (Nodes: ${best.nodes.length})`, "success");
            } else {
                const frontSize = evaluatedPop.filter(p => p.paretoRank === 0).length;
                addLog(`Generation ${generation} complete. Pareto front: ${frontSize} solutions.`, "success");
            }

            // Update Hall of Fame (keep top 5 overall) - SINGLE-OBJECTIVE MODE
            if (!settings.useMultiObjective) {
                setHallOfFame(prev => {
                    const combined = [...prev, best].sort((a, b) => (b.adjustedFitness || 0) - (a.adjustedFitness || 0));
                    const unique = Array.from(new Map(combined.map(item => [item.id, item])).values());
                    return unique.slice(0, 5);
                });
            }
            
            // === PARETO ARCHIVE (Multi-Objective Mode) ===
            if (settings.useMultiObjective) {
                setParetoArchive(prev => {
                    // Combine with current evaluated population
                    const combined = [...prev, ...evaluatedPop];
                    
                    // Sort by Pareto ranking
                    const paretoItems = combined.map(p => ({ id: p.id, vector: p.objectives! }));
                    const rankings = nonDominatedSorting(paretoItems);
                    
                    // Get front 0 (non-dominated solutions)
                    const front0 = combined.filter(p => rankings.get(p.id)!.rank === 0);
                    
                    // Sort by crowding distance (diverse first)
                    front0.sort((a, b) => 
                        (rankings.get(b.id)!.crowdingDistance || 0) - 
                        (rankings.get(a.id)!.crowdingDistance || 0)
                    );
                    
                    // Limit archive size
                    return front0.slice(0, settings.paretoFrontSize);
                });
            }

            // Update stats history
            const avgNodes = Math.round(evaluatedPop.reduce((acc, p) => acc + p.nodes.length, 0) / evaluatedPop.length);
            setStats(prev => [...prev, {
                generation,
                bestFitness: best.adjustedFitness || 0,
                avgNodes
            }]);

            // Finalize the current generation's snapshot with fitness scores
            setGenerationHistory(prev => {
                if (prev.length === 0) return prev;
                const updated = [...prev];
                updated[updated.length - 1] = {
                    generation,
                    genomes: [...evaluatedPop],
                    bestFitness: best.adjustedFitness || 0,
                    avgNodes,
                    timestamp: new Date().toLocaleTimeString(),
                    evaluated: true
                };
                return updated;
            });

            // Clear per-genome metrics for next generation
            perGenomeMetricsRef.current = new Map();

            // 5. Check if we should stop
            if (!isRunningRef.current) return;

            // === MULTI-OBJECTIVE: Convergence-based stopping ===
            if (settings.useMultiObjective && settings.stoppingCriteria.useHypervolumeConvergence) {
                // Initialize checker if needed
                if (!convergenceCheckerRef.current) {
                    convergenceCheckerRef.current = new ConvergenceChecker(settings.stoppingCriteria);
                }
                
                // Get Pareto front (rank 0)
                const paretoFront = evaluatedPop
                    .filter(p => p.paretoRank === 0)
                    .map(p => ({ id: p.id, vector: p.objectives! }));
                
                // Check convergence
                const report = convergenceCheckerRef.current.checkConvergence(paretoFront);
                
                if (report.shouldStop && report.reason) {
                    addLog(
                        `Evolution converged: ${report.reason}. ` +
                        `Hypervolume: ${report.metrics.hypervolume.toFixed(4)}, ` +
                        `Improvement: ${report.metrics.hypervolumeImprovement.toFixed(6)}, ` +
                        `Generations without improvement: ${report.metrics.generationsWithoutImprovement}, ` +
                        `Pareto front size: ${report.metrics.frontSize}`,
                        "success"
                    );
                    setIsRunning(false);
                    isRunningRef.current = false;
                    invoke('stop_evolution').catch(() => { });
                    return;
                }
            }

            // Max generations auto-stop (existing logic)
            if (settings.useMaxGenerations && generation + 1 >= settings.maxGenerations) {
                addLog(`Reached max generations limit (${settings.maxGenerations}). Stopping evolution.`, "warn");
                setIsRunning(false);
                isRunningRef.current = false;
                invoke('stop_evolution').catch(() => { });
                return;
            }

            // 6. Selection, Crossover, Mutation (Breed next generation)
            addLog(`Breeding Generation ${generation + 1}...`);
            const popSize = evaluatedPop.length;
            const nextGen: PopulatedGenome[] = [];

            // Elitism (Keep top 10%)
            const eliteCount = Math.max(1, Math.floor(popSize * 0.10));
            for (let i = 0; i < eliteCount; i++) {
                nextGen.push(evaluatedPop[i]);
            }

            // === SELECTION ===
            // Simple Tournament Selection (single-objective)
            const tournamentSelect = () => {
                const i1 = Math.floor(Math.random() * popSize);
                const i2 = Math.floor(Math.random() * popSize);
                return (evaluatedPop[i1].adjustedFitness || 0) > (evaluatedPop[i2].adjustedFitness || 0) ? evaluatedPop[i1] : evaluatedPop[i2];
            };
            
            // Pareto Tournament Selection (multi-objective)
            const tournamentSelectPareto = () => {
                const i1 = Math.floor(Math.random() * popSize);
                const i2 = Math.floor(Math.random() * popSize);
                const p1 = evaluatedPop[i1];
                const p2 = evaluatedPop[i2];
                
                // Compare by Pareto rank first (lower is better)
                const rank1 = p1.paretoRank || 0;
                const rank2 = p2.paretoRank || 0;
                
                if (rank1 !== rank2) {
                    return rank1 < rank2 ? p1 : p2;
                }
                
                // Same rank: compare by crowding distance (higher is better for diversity)
                const dist1 = p1.crowdingDistance || 0;
                const dist2 = p2.crowdingDistance || 0;
                
                return dist1 > dist2 ? p1 : p2;
            };

            const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

            let breedAttempts = 0;
            while (nextGen.length < popSize && breedAttempts < popSize * 10) {
                breedAttempts++;
                try {
                    // Select parents based on mode
                    const selectParent = settings.useMultiObjective ? tournamentSelectPareto : tournamentSelect;
                    const parentA = selectParent()!.genome;
                    const parentB = selectParent()!.genome;

                    // Crossover
                    let childGenome: Genome | null = null;
                    const activeStrategies = settings.selectedCrossovers.filter(s =>
                        s === 'subgraph-insertion' || s === 'subgraph-replacement' || s === 'neat-style' || s === 'multi-point'
                    );

                    if (activeStrategies.length > 0) {
                        const strategy = activeStrategies[Math.floor(Math.random() * activeStrategies.length)];
                        try {
                            if (strategy === 'subgraph-replacement') {
                                const res = parentA.BreedByReplacement(parentB, maxNodes);
                                if (res) childGenome = res.genome;
                            } else if (strategy === 'neat-style') {
                                const res = parentA.BreedNeatStyle(parentB, maxNodes);
                                if (res) childGenome = res.genome;
                            } else if (strategy === 'multi-point') {
                                const res = parentA.BreedMultiPoint(parentB, maxNodes);
                                if (res) childGenome = res.genome;
                            } else {
                                // Default 'subgraph-insertion'
                                const res = parentA.Breed(parentB, maxNodes);
                                if (res) childGenome = res.genome;
                            }
                        } catch (e) {
                            // silently fail crossover and fallback
                        }
                    }

                    // Fallback to clone if crossover failed or was disabled
                    if (!childGenome) {
                        const parentAStr = await serializeGenome(parentA);
                        const { genome: clone } = await deserializeGenome(parentAStr);
                        childGenome = clone;
                    }

                    // Mutation
                    const dynamicRates = settings.useAdaptiveMutation
                        ? getAdaptiveMutationRates(childGenome.getAllNodes().length)
                        : {
                            addNode: settings.mutationRates.addNode,
                            removeNode: settings.mutationRates.removeNode,
                            removeSubgraph: settings.mutationRates.removeSubgraph
                        };

                    if (Math.random() < dynamicRates.removeSubgraph) {
                        const res = childGenome.MutateRemoveSubgraph();
                        if (res) childGenome = res.genome;
                    }
                    if (Math.random() < dynamicRates.removeNode) {
                        const res = childGenome.MutateRemoveNode();
                        if (res) childGenome = res.genome;
                    }
                    if (Math.random() < dynamicRates.addNode) {
                        const res = childGenome.MutateAddNode(maxNodes);
                        if (res) childGenome = res.genome;
                    }
                    if (settings.mutationRates.addSkipConnection && Math.random() < settings.mutationRates.addSkipConnection) {
                        const res = childGenome.MutateAddSkipConnection(maxNodes);
                        if (res) childGenome = res.genome;
                    }
                    if (settings.mutationRates.changeLayerType && Math.random() < settings.mutationRates.changeLayerType) {
                        const res = childGenome.MutateChangeLayerType(maxNodes);
                        if (res) childGenome = res.genome;
                    }

                    // Params mutation
                    if (settings.mutationRates.params && Math.random() < settings.mutationRates.params) {
                        const mutationOpts = new Map<string, number>();
                        mutationOpts.set('params', settings.mutationRates.params);
                        childGenome.getAllNodes().forEach(node => {
                            if (typeof (node as any).Mutate === 'function') {
                                (node as any).Mutate(mutationOpts);
                            }
                        });
                    }

                    const nextGenNodes = childGenome.getAllNodes();
                    if (!Genome.isGenomeFeasible(nextGenNodes)) {
                        if (breedAttempts % 10 === 0) {
                            addLog(`Rejected invalid child (shape mismatch). Retrying breeding...`, "warn");
                        }
                        continue;
                    }

                    nextGen.push({
                        id: crypto.randomUUID(),
                        genome: childGenome,
                        nodes: nextGenNodes
                    });
                } catch (e) {
                    console.warn(`[Breed] Child generation failed (likely shape mismatch). Attempt ${breedAttempts}/${popSize * 10}`, e);
                    continue;
                }
            }

            setPopulation(nextGen);
            setGeneration(g => g + 1);

            // Save pre-eval snapshot for the new generation immediately
            const nextGenNum = generation + 1;
            setGenerationHistory(prev => [...prev, {
                generation: nextGenNum,
                genomes: [...nextGen],
                bestFitness: 0,
                avgNodes: Math.round(nextGen.reduce((acc, g) => acc + g.nodes.length, 0) / nextGen.length),
                timestamp: new Date().toLocaleTimeString(),
                evaluated: false
            }]);

            // Timeout to yield to React rendering before next loop iteration
            setTimeout(() => {
                if (isRunningRef.current) {
                    runGeneration();
                }
            }, 100);

        } catch (err) {
            console.error('evaluate_population error:', err);
            addLog(`Evaluation failed: ${String(err)}`, "error");
            // Don't auto-stop - let user manually click Stop to investigate
            // stopEvolution();
        }

    }, [datasetProfileId, generation, population, settings, addLog, stopEvolution]);

    const startEvolution = useCallback((seedGenomes: string[]) => {
        try {
            if (!datasetProfileId) {
                addLog("Cannot start: No dataset selected!", "error");
                return;
            }

            if (seedGenomes.length === 0 && !settings.useRandomInitialization) {
                addLog("Cannot start: No seeds provided and Random Initialization disabled!", "error");
                return;
            }

            // === PHASE 3: Validate dataset profile before evolution ===
            const profiles = useDatasetManagerStore.getState().profiles;
            const profile = profiles.find(p => p.id === datasetProfileId);
            
            if (profile) {
                // Check if dataset has been scanned and validated
                if (!profile.isScanned) {
                    addLog(
                        "Cannot start: Dataset has not been scanned. " +
                        "Go to Dataset Manager, select the dataset, and click 'Scan Dataset'.",
                        "error"
                    );
                    return;
                }

                // Check if dataset validation passed
                if (!profile.isValidForEvolution) {
                    const issues = profile.validationReport?.issues || [];
                    const issueMessages = issues
                        .map(issue => `${issue.component}: ${issue.message}`)
                        .join('; ');
                    
                    addLog(
                        `Cannot start: Dataset validation failed. Issues: ${issueMessages || 'Unknown validation error'}. ` +
                        `Please fix issues in Dataset Manager before starting evolution.`,
                        "error"
                    );
                    return;
                }

                // Log dataset validation success
                if (profile.validationReport) {
                    const inputShapesObj = profile.validationReport.input_shapes || {};
                    const inputShapes = Object.entries(inputShapesObj)
                        .map(([streamId, shape]) => `${streamId}: [${(shape || []).join(',')}]`)
                        .join('; ');
                    const outputShape = profile.validationReport.output_shape || [];
                    addLog(
                        `✓ Dataset validated. Input shapes: ${inputShapes}. Output shape: [${outputShape.join(',')}]`,
                        "success"
                    );
                }
            }

            // Validate dataset percentage vs splits
            if (profile) {
                const totalSamples = profile.totalSamples || profile.scanResult?.totalMatched || 0;
                const usedSamples = Math.floor((totalSamples * (settings.datasetPercent || 100)) / 100);
                const { train, val, test } = profile.split;
                const splitSum = train + val + test;

                if (splitSum > 0 && usedSamples > 0) {
                    const trainCount = Math.floor((usedSamples * train) / splitSum);
                    const valCount = Math.floor((usedSamples * val) / splitSum);
                    const testCount = Math.floor((usedSamples * test) / splitSum);

                    const errors: string[] = [];
                    if (train > 0 && trainCount < 1) errors.push(`Train (${train}%): 0 samples`);
                    if (val > 0 && valCount < 1) errors.push(`Validation (${val}%): 0 samples`);
                    if (test > 0 && testCount < 1) errors.push(`Test (${test}%): 0 samples`);

                    if (errors.length > 0) {
                        addLog(
                            `Cannot start: Dataset percentage ${settings.datasetPercent}% (${usedSamples} samples) ` +
                            `is too low for the configured splits. ${errors.join('; ')}. ` +
                            `Increase Dataset Usage % or adjust splits in Dataset Manager.`,
                            "error"
                        );
                        return;
                    }

                    addLog(
                        `Dataset: ${usedSamples} samples → Train: ${trainCount}, Val: ${valCount}, Test: ${testCount}`,
                        "info"
                    );
                }
            }

            setIsRunning(true);
            isRunningRef.current = true;

            initPopulation(seedGenomes);
        } catch (err: any) {
            console.error("startEvolution error:", err);
            addLog(`Error during startEvolution: ${err.message || String(err)}`, "error");
        }
    }, [datasetProfileId, addLog, initPopulation, settings]);

    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => {
        if (isRunning && population.length > 0) {
            // Need a slight delay to allow React rendering/logging to flush
            const timer = setTimeout(() => {
                runGeneration();
            }, 100);
            return () => clearTimeout(timer);
        }
        return;
    }, [isRunning, population, runGeneration]);

    return {
        isRunning,
        startEvolution,
        stopEvolution,
        generation,
        population,
        hallOfFame,
        paretoArchive,  // New: Pareto Archive for multi-objective mode
        logs,
        stats,
        runGeneration,
        currentEvaluatingIndex,
        liveMetrics,
        generationHistory
    };
};
