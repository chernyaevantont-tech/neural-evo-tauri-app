import { useState, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useEvolutionSettingsStore, getAdaptiveMutationRates } from '../../evolution-manager/model/store';
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

    // Initial Spawning (from a base JSON or random)
    const initPopulation = useCallback(async (baseGraphJSON: string, popSize: number = 20) => {
        const genomes: PopulatedGenome[] = [];
        try {
            for (let i = 0; i < popSize; i++) {
                // For now, duplicate the base graph.
                const { genome: newGenome, nodes } = await deserializeGenome(baseGraphJSON);
                genomes.push({
                    id: crypto.randomUUID(),
                    nodes: nodes,
                    genome: newGenome
                });
            }
            setPopulation(genomes);
            setGeneration(0);
            setStats([]);
            addLog(`Spawned Generation 0 with ${popSize} identical seeds.`, "info");
        } catch (e) {
            addLog(`Failed to initialize population: ${String(e)}`, "error");
        }
    }, [addLog]);

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

            // 2. Call Rust Evaluator
            const results = await invoke<EvaluationResult[]>('evaluate_population', {
                genomes: serializedGenomes,
                datasetProfile: datasetProfileId
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

            while (nextGen.length < popSize) {
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
                        } else {
                            const res = parentA.Breed(parentB, maxNodes);
                            if (res) childGenome = res.genome;
                        }
                    } catch (e) {
                        // silently fail crossover and fallback
                    }
                }

                // Fallback to clone if crossover failed
                if (!childGenome) {
                    const parentAStr = await serializeGenome(parentA);
                    const { genome: clone } = await deserializeGenome(parentAStr);
                    childGenome = clone;
                }

                // Mutation
                let mutated = false;
                const dynamicRates = settings.useAdaptiveMutation
                    ? getAdaptiveMutationRates(childGenome.getAllNodes().length)
                    : {
                        addNode: settings.mutationRates.addNode,
                        removeNode: settings.mutationRates.removeNode,
                        removeSubgraph: settings.mutationRates.removeSubgraph
                    };

                if (Math.random() < dynamicRates.removeSubgraph) {
                    const res = childGenome.MutateRemoveSubgraph();
                    if (res) mutated = true;
                }
                if (!mutated && Math.random() < dynamicRates.removeNode) {
                    const res = childGenome.MutateRemoveNode();
                    if (res) mutated = true;
                }
                if (!mutated && Math.random() < dynamicRates.addNode) {
                    const res = childGenome.MutateAddNode(maxNodes);
                    if (res) mutated = true;
                }

                nextGen.push({
                    id: crypto.randomUUID(),
                    genome: childGenome,
                    nodes: childGenome.getAllNodes()
                });
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

    const startEvolution = useCallback((baseJSON?: string) => {
        if (!datasetProfileId) {
            addLog("Cannot start: No dataset selected!", "error");
            return;
        }
        setIsRunning(true);
        isRunningRef.current = true;

        if (baseJSON) {
            initPopulation(baseJSON);
            // The generation loop will trigger via a useEffect watching population/isRunning later
        }
    }, [datasetProfileId, addLog, initPopulation]);

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
