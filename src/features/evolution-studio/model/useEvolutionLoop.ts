import { useState, useCallback, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useEvolutionSettingsStore, getAdaptiveMutationRates } from '../../evolution-manager/model/store';
import { useDatasetManagerStore } from '../../../features/dataset-manager/model/store';
import { Genome, BaseNode, serializeGenome, deserializeGenome } from '../../../entities/canvas-genome';

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
    adjustedFitness?: number; // Base fitness penalized by bloat
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

export const useEvolutionLoop = (datasetProfileId: string | null) => {
    const settings = useEvolutionSettingsStore();

    const [isRunning, setIsRunning] = useState(false);
    const [generation, setGeneration] = useState(0);
    const [population, setPopulation] = useState<PopulatedGenome[]>([]);
    const [hallOfFame, setHallOfFame] = useState<PopulatedGenome[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [stats, setStats] = useState<GenerationStat[]>([]);

    // Using refs for safe async access within loops
    const isRunningRef = useRef(false);

    const addLog = useCallback((msg: string, type: LogEntry['type'] = 'info') => {
        setLogs(prev => [...prev.slice(-49), { time: new Date().toLocaleTimeString(), message: msg, type }]);
    }, []);

    const stopEvolution = useCallback(() => {
        setIsRunning(false);
        isRunningRef.current = false;
        addLog("Evolution stopped by user.", "warn");
    }, [addLog]);

    // Initial Spawning (from multiple library seeds or a fallback graph)
    const initPopulation = useCallback(async (seedJSONs: string[], popSize: number = 20) => {
        const genomes: PopulatedGenome[] = [];
        try {
            // First pass: instantiate all selected seeds
            const seedInstances = [];
            for (const seedStr of seedJSONs) {
                const { genome, nodes } = await deserializeGenome(seedStr);
                seedInstances.push(genome);
                genomes.push({
                    id: crypto.randomUUID(),
                    nodes: nodes,
                    genome: genome
                });
            }

            // Fallback if no seeds provided (shouldn't happen with UI checks, but just in case)
            if (seedInstances.length === 0) {
                addLog("No seeds provided, initialization aborted.", "error");
                return;
            }

            // Fill the rest of the population up to popSize by mutating the seeds
            let initAttempts = 0;
            while (genomes.length < popSize && initAttempts < popSize * 10) {
                initAttempts++;
                try {
                    // Pick a random seed
                    const baseSeed = seedInstances[Math.floor(Math.random() * seedInstances.length)];

                    // Clone the seed
                    const seedStr = await serializeGenome(baseSeed);
                    let { genome: clone } = await deserializeGenome(seedStr);

                    // Heavily mutate the clone to build initial diversity
                    // Apply a few rounds of mutation to drift away from the seed
                    const mutationRounds = Math.floor(Math.random() * 3) + 1;
                    const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

                    for (let r = 0; r < mutationRounds; r++) {
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

                        // Params mutation
                        if (settings.mutationRates.params && Math.random() < settings.mutationRates.params) {
                            const mutationOpts = new Map<string, number>();
                            mutationOpts.set('params', settings.mutationRates.params);
                            clone.getAllNodes().forEach(node => {
                                if (typeof (node as any).Mutate === 'function') {
                                    (node as any).Mutate(mutationOpts);
                                }
                            });
                        }
                    }

                    genomes.push({
                        id: crypto.randomUUID(),
                        genome: clone,
                        nodes: clone.getAllNodes()
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
            addLog(`Spawned Generation 0: ${seedJSONs.length} direct seeds, ${popSize - seedJSONs.length} mutated clones.`, "info");
        } catch (e) {
            addLog(`Failed to initialize population: ${String(e)}`, "error");
        }
    }, [addLog, settings]);

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

            // Look up dataset split percentages from the dataset manager
            const profiles = useDatasetManagerStore.getState().profiles;
            const currentProfile = profiles.find(p => p.id === datasetProfileId);
            const trainSplit = currentProfile?.split?.train ?? 80;
            const valSplit = currentProfile?.split?.val ?? 10;
            const testSplit = currentProfile?.split?.test ?? 10;

            // 2. Call Rust Evaluator
            const results = await invoke<EvaluationResult[]>('evaluate_population', {
                genomes: serializedGenomes,
                datasetProfile: datasetProfileId,
                batchSize: settings.batchSize || 32,
                evalEpochs: settings.evalEpochs || 1,
                datasetPercent: settings.datasetPercent || 100,
                trainSplit,
                valSplit,
                testSplit,
            });

            // 3. Map Results & Apply Parsimony Pressure (Bloat Control)
            const alpha = settings.useParsimonyPressure ? settings.parsimonyAlpha : 0;
            const evaluatedPop = population.map((p, index) => {
                const res = results[index];
                const nodeCount = p.genome.getAllNodes().length;

                // Let's assume fitness is inversely proportional to Loss (higher fitness is better).
                // Base Fitness = (1.0 - Loss) or just -Loss.
                // We will use Accuracy for fitness if available, or (1 / (1 + Loss))
                const baseFitness = res.accuracy > 0 ? res.accuracy : (1 / (1 + res.loss));

                return {
                    ...p,
                    loss: res.loss,
                    accuracy: res.accuracy,
                    adjustedFitness: baseFitness - (alpha * nodeCount)
                };
            });

            // 4. Sort by Adjusted Fitness (descending)
            evaluatedPop.sort((a, b) => (b.adjustedFitness || 0) - (a.adjustedFitness || 0));

            const best = evaluatedPop[0];
            addLog(`Generation ${generation} complete. Best Fitness: ${best.adjustedFitness?.toFixed(4)} (Nodes: ${best.nodes.length})`, "success");

            // Update Hall of Fame (keep top 5 overall)
            setHallOfFame(prev => {
                const combined = [...prev, best].sort((a, b) => (b.adjustedFitness || 0) - (a.adjustedFitness || 0));
                // Ensure uniqueness by ID or hash
                const unique = Array.from(new Map(combined.map(item => [item.id, item])).values());
                return unique.slice(0, 5);
            });

            // Update stats history
            const avgNodes = Math.round(evaluatedPop.reduce((acc, p) => acc + p.nodes.length, 0) / evaluatedPop.length);
            setStats(prev => [...prev, {
                generation,
                bestFitness: best.adjustedFitness || 0,
                avgNodes
            }]);

            // 5. Check if we should stop
            if (!isRunningRef.current) return;

            // 6. Selection, Crossover, Mutation (Breed next generation)
            addLog(`Breeding Generation ${generation + 1}...`);
            const popSize = evaluatedPop.length;
            const nextGen: PopulatedGenome[] = [];

            // Elitism (Keep top 10%)
            const eliteCount = Math.max(1, Math.floor(popSize * 0.10));
            for (let i = 0; i < eliteCount; i++) {
                nextGen.push(evaluatedPop[i]);
            }

            // Simple Tournament Selection
            const tournamentSelect = () => {
                const i1 = Math.floor(Math.random() * popSize);
                const i2 = Math.floor(Math.random() * popSize);
                return (evaluatedPop[i1].adjustedFitness || 0) > (evaluatedPop[i2].adjustedFitness || 0) ? evaluatedPop[i1] : evaluatedPop[i2];
            };

            const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

            let breedAttempts = 0;
            while (nextGen.length < popSize && breedAttempts < popSize * 10) {
                breedAttempts++;
                try {
                    const parentA = tournamentSelect()!.genome;
                    const parentB = tournamentSelect()!.genome;

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

                    nextGen.push({
                        id: crypto.randomUUID(),
                        genome: childGenome,
                        nodes: childGenome.getAllNodes()
                    });
                } catch (e) {
                    console.warn(`[Breed] Child generation failed (likely shape mismatch). Attempt ${breedAttempts}/${popSize * 10}`, e);
                    continue;
                }
            }

            setPopulation(nextGen);
            setGeneration(g => g + 1);

            // Timeout to yield to React rendering before next loop iteration
            setTimeout(() => {
                if (isRunningRef.current) {
                    runGeneration();
                }
            }, 100);

        } catch (err) {
            console.error(err);
            addLog(`Evolution Error: ${String(err)}`, "error");
            stopEvolution();
        }

    }, [datasetProfileId, generation, population, settings, addLog, stopEvolution]);

    const startEvolution = useCallback((seedGenomes: string[]) => {
        if (!datasetProfileId) {
            addLog("Cannot start: No dataset selected!", "error");
            return;
        }

        if (seedGenomes.length === 0) {
            addLog("Cannot start: No seeds provided!", "error");
            return;
        }

        // Validate dataset percentage vs splits
        const profiles = useDatasetManagerStore.getState().profiles;
        const profile = profiles.find(p => p.id === datasetProfileId);
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
    }, [datasetProfileId, addLog, initPopulation, settings]);

    useEffect(() => {
        if (isRunning && population.length > 0) {
            // Need a slight delay to allow React rendering/logging to flush
            const timer = setTimeout(() => {
                runGeneration();
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [isRunning, population, runGeneration]);

    return {
        isRunning,
        startEvolution,
        stopEvolution,
        generation,
        population,
        hallOfFame,
        logs,
        stats,
        runGeneration // Exposed to be called manually or via useEffect
    };
};
