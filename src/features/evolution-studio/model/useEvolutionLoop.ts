import { useState, useCallback, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Genome, serializeGenome, deserializeGenome, generateRandomArchitecture, extractShapesFromDatasetProfile } from '../../../entities/canvas-genome';
import type { BatchMetrics, GenerationSnapshot, PopulatedGenome } from '../../../entities/genome';
import type {
    AdaptiveMutationSettings,
    GenomeGenealogy,
    MutationType,
    TrainingProfiler,
    UseEvolutionLoopParams,
} from '../../../shared/lib';
import { computeZeroCostScore, ZeroCostMetrics } from './useZeroCostEvaluation';

const DEFAULT_MAX_SINGLE_TENSOR_MB = 192;
const DEFAULT_MAX_WORKING_SET_MB = 2048;
const TRAINING_WORKING_SET_MULTIPLIER = 4;
const MB_TO_BYTES = 1024 * 1024;
const MIN_WORKING_SET_BYTES = 8 * MB_TO_BYTES;
const INIT_ATTEMPT_MULTIPLIER = 30;
const BREED_ATTEMPT_MULTIPLIER = 30;
const STRUCTURAL_RETRY_ATTEMPTS = 30;

type MemoryValidation = {
    ok: boolean;
    reason?: string;
};

function validateGenomeTensorBudget(
    genome: Genome,
    batchSize: number,
    maxSingleTensorBytes: number,
    maxWorkingSetBytes: number,
): MemoryValidation {
    const nodes = genome.getAllNodes();
    let maxPerSampleTensorBytes = 0;
    let totalPerSampleActivationBytes = 0;

    for (const node of nodes) {
        const shape = node.GetOutputShape();

        if (!Array.isArray(shape) || shape.length === 0) {
            continue;
        }

        let elements = 1;
        for (const dim of shape) {
            if (!Number.isFinite(dim) || dim <= 0 || !Number.isInteger(dim)) {
                return {
                    ok: false,
                    reason: `invalid tensor shape at ${node.GetNodeType()} (${JSON.stringify(shape)})`
                };
            }

            elements *= dim;
            if (!Number.isFinite(elements) || elements > Number.MAX_SAFE_INTEGER) {
                return {
                    ok: false,
                    reason: `tensor elements overflow at ${node.GetNodeType()} (${JSON.stringify(shape)})`
                };
            }
        }

        const perSampleTensorBytes = elements * 4; // f32 activations
        if (perSampleTensorBytes > maxPerSampleTensorBytes) {
            maxPerSampleTensorBytes = perSampleTensorBytes;
        }
        totalPerSampleActivationBytes += perSampleTensorBytes;

        if (!Number.isFinite(totalPerSampleActivationBytes) || totalPerSampleActivationBytes > Number.MAX_SAFE_INTEGER) {
            return {
                ok: false,
                reason: 'activation memory estimate overflowed'
            };
        }
    }

    const effectiveBatch = Math.max(1, Math.floor(batchSize || 1));
    const estimatedSingleTensorBatchBytes = maxPerSampleTensorBytes * effectiveBatch;
    if (estimatedSingleTensorBatchBytes > maxSingleTensorBytes) {
        return {
            ok: false,
            reason:
                `single batched tensor too large: ${(estimatedSingleTensorBatchBytes / (1024 * 1024)).toFixed(1)}MB ` +
                `(limit ${(maxSingleTensorBytes / (1024 * 1024)).toFixed(1)}MB)`
        };
    }

    // Approximate training memory: activations for forward/backward plus optimizer/work buffers.
    const estimatedWorkingSetBytes = totalPerSampleActivationBytes * effectiveBatch * TRAINING_WORKING_SET_MULTIPLIER;
    if (estimatedWorkingSetBytes > maxWorkingSetBytes) {
        return {
            ok: false,
            reason:
                `estimated working set too large: ${(estimatedWorkingSetBytes / (1024 * 1024)).toFixed(1)}MB ` +
                `(limit ${(maxWorkingSetBytes / (1024 * 1024)).toFixed(1)}MB)`
        };
    }

    return { ok: true };
}

export interface EvaluationResult {
    genome_id: string;
    loss: number;
    accuracy: number;
    profiler?: TrainingProfiler;
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

export interface GenomeResultEvent {
    index: number;
    loss: number;
    accuracy: number;
}

function getAdaptiveMutationRates(settings: AdaptiveMutationSettings, currentNodes: number) {
    if (!settings.useAdaptiveMutation) {
        return {
            addNode: settings.mutationRates.addNode,
            removeNode: settings.mutationRates.removeNode,
            removeSubgraph: settings.mutationRates.removeSubgraph,
        };
    }

    const target = settings.adaptiveTargetNodes;
    if (currentNodes <= target) {
        const ratio = currentNodes / Math.max(1, target);
        return {
            addNode: Math.max(0.1, 0.4 - 0.2 * ratio),
            removeNode: Math.max(0.01, 0.05 * ratio),
            removeSubgraph: Math.max(0.01, 0.02 * ratio),
        };
    }

    const ratio = Math.min(2.0, currentNodes / target);
    return {
        addNode: Math.max(0.01, 0.2 - 0.1 * ratio),
        removeNode: Math.min(0.8, 0.1 + 0.3 * (ratio - 1)),
        removeSubgraph: Math.min(0.5, 0.05 + 0.2 * (ratio - 1)),
    };
}

export const useEvolutionLoop = ({ datasetProfileId, settings, datasetProfiles }: UseEvolutionLoopParams) => {

    const [isRunning, setIsRunning] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [generation, setGeneration] = useState(0);
    const [population, setPopulation] = useState<PopulatedGenome[]>([]);
    const [hallOfFame, setHallOfFame] = useState<PopulatedGenome[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [stats, setStats] = useState<GenerationStat[]>([]);

    const [currentEvaluatingIndex, setCurrentEvaluatingIndex] = useState<number>(0);
    const [liveMetrics, setLiveMetrics] = useState<BatchMetrics[]>([]);
    const [generationHistory, setGenerationHistory] = useState<GenerationSnapshot[]>([]);

    // Per-genome metrics accumulator (ref to avoid re-renders on every batch)
    const perGenomeMetricsRef = useRef<Map<number, BatchMetrics[]>>(new Map());
    const activeGenomeIndexRef = useRef<number>(0);
    const genealogyMapRef = useRef<Map<string, GenomeGenealogy>>(new Map());
    const generationRunInFlightRef = useRef(false);

    // Using refs for safe async access within loops
    const isRunningRef = useRef(false);
    const isPausedRef = useRef(false);

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
                activeGenomeIndexRef.current = event.payload;
                perGenomeMetricsRef.current.set(event.payload, []);
                setLiveMetrics(prev => prev.filter(m => m.genome_index !== event.payload));
                // Add log for the user
                addLog(`Starting evaluation for Genome #${event.payload + 1}...`, 'info');
            }).then(fn => {
                unlistenStart = fn;
            });

            listen<BatchMetrics>('evaluating-batch-metrics', (event) => {
                setLiveMetrics(prev => {
                    const next = [...prev, event.payload];
                    return next.length > 3000 ? next.slice(next.length - 3000) : next;
                });
                // Also store per-genome
                const idx = event.payload.genome_index ?? activeGenomeIndexRef.current;
                const arr = perGenomeMetricsRef.current.get(idx);
                if (arr) arr.push(event.payload);
                else perGenomeMetricsRef.current.set(idx, [event.payload]);
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

    const registerGenealogyEvent = useCallback(async (
        genomeId: string,
        generationNum: number,
        parentIds: string[],
        mutationType: MutationType,
        mutationParams: Record<string, unknown>,
    ) => {
        if (!settings.genealogyTrackingEnabled) {
            return;
        }

        const record: GenomeGenealogy = {
            genome_id: genomeId,
            generation: generationNum,
            parent_ids: parentIds,
            mutation_type: mutationType,
            mutation_params: mutationParams,
            fitness: 0,
            accuracy: 0,
            created_at_ms: Date.now(),
        };
        genealogyMapRef.current.set(genomeId, record);

        try {
            if (parentIds.length === 0) {
                await invoke('register_founder', { genomeId, generation: generationNum });
                return;
            }

            if (mutationType.type === 'Crossover' && parentIds.length >= 2) {
                await invoke('register_crossover', {
                    parentA: parentIds[0],
                    parentB: parentIds[1],
                    childId: genomeId,
                    generation: generationNum,
                });
                return;
            }

            await invoke('register_mutation', {
                parentId: parentIds[0],
                childId: genomeId,
                mutationType,
                generation: generationNum,
            });
        } catch (e) {
            addLog(`Genealogy backend sync failed for ${genomeId}: ${String(e)}`, 'warn');
        }
    }, [addLog, settings.genealogyTrackingEnabled]);

    const getGenomeBudgetValidator = useCallback(() => {
        // Always derive mutation-time memory checks from the active target device budget from UI.
        const customRamBytes = Math.max(
            0,
            Math.floor((settings.customDeviceParams?.ram_mb ?? 0) * MB_TO_BYTES),
        );
        const configuredRamBytes = Math.max(0, Math.floor(settings.resourceTargets.ram));
        const targetRamBytes = customRamBytes > 0
            ? customRamBytes
            : configuredRamBytes > 0
                ? configuredRamBytes
                : DEFAULT_MAX_WORKING_SET_MB * MB_TO_BYTES;
        const safetyMarginBytes = Math.max(
            0,
            Math.floor((settings.memorySafetyMarginMb ?? 0) * MB_TO_BYTES),
        );
        const estimatorSafetyFactor = Math.max(1, settings.estimatorSafetyFactor ?? 1);

        const availableBytes = Math.max(MIN_WORKING_SET_BYTES, targetRamBytes - safetyMarginBytes);
        const maxWorkingSetBytes = Math.max(
            MIN_WORKING_SET_BYTES,
            Math.floor(availableBytes / estimatorSafetyFactor),
        );

        // Keep a per-tensor cap as a fraction of the remaining device budget.
        const maxSingleTensorBytes = Math.max(
            1 * MB_TO_BYTES,
            Math.min(
                maxWorkingSetBytes,
                Math.floor(maxWorkingSetBytes * 0.35),
                DEFAULT_MAX_SINGLE_TENSOR_MB * MB_TO_BYTES,
            ),
        );

        return (genome: Genome): MemoryValidation => validateGenomeTensorBudget(
            genome,
            settings.batchSize || 32,
            maxSingleTensorBytes,
            maxWorkingSetBytes,
        );
    }, [
        settings.batchSize,
        settings.customDeviceParams?.ram_mb,
        settings.resourceTargets.ram,
        settings.memorySafetyMarginMb,
        settings.estimatorSafetyFactor,
    ]);

    const stopEvolution = useCallback(() => {
        setIsRunning(false);
        setIsPaused(false);
        isRunningRef.current = false;
        isPausedRef.current = false;
        // Signal the Rust backend to stop the current evaluation pass
        invoke('stop_evolution').catch(err => console.error('Failed to stop backend:', err));
        addLog("Evolution stopped by user.", "warn");
    }, [addLog]);

    const pauseEvolution = useCallback(() => {
        if (!isRunningRef.current || isPausedRef.current) {
            return;
        }

        setIsPaused(true);
        isPausedRef.current = true;
        isRunningRef.current = false;
        addLog('Pause requested. Current generation will finish, then loop will halt.', 'warn');
    }, [addLog]);

    const resumeEvolution = useCallback(() => {
        if (!isRunning || !isPausedRef.current) {
            return;
        }

        setIsPaused(false);
        isPausedRef.current = false;
        isRunningRef.current = true;
        addLog('Evolution resumed.', 'success');
    }, [addLog, isRunning]);

    const saveCheckpoint = useCallback(async () => {
        const snapshotPayload = {
            savedAt: new Date().toISOString(),
            generation,
            isRunning,
            isPaused,
            hallOfFameIds: hallOfFame.map((g) => g.id),
            populationIds: population.map((g) => g.id),
            generationHistorySize: generationHistory.length,
        };

        try {
            localStorage.setItem('evolution-runtime-checkpoint', JSON.stringify(snapshotPayload));
            addLog(`Checkpoint saved for generation ${generation}.`, 'success');
        } catch (e) {
            addLog(`Checkpoint save failed: ${String(e)}`, 'error');
        }
    }, [addLog, generation, generationHistory.length, hallOfFame, isPaused, isRunning, population]);

    // Initial Spawning (from multiple library seeds or a fallback graph)
    const initPopulation = useCallback(async (seedJSONs: string[]) => {
        const popSize = settings.populationSize;
        const genomes: PopulatedGenome[] = [];
        const validateBudget = getGenomeBudgetValidator();
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
                const profile = datasetProfiles.find(p => p.id === datasetProfileId);

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
                while (genomes.length < numRandom && randomAttempts < numRandom * 12) {
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

                        const randomBudget = validateBudget(randomGenome);
                        if (!randomBudget.ok) {
                            addLog(`Random architecture rejected by memory budget: ${randomBudget.reason}`, "warn");
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

                        const finalRandomNodes = randomGenome.getAllNodes();
                        if (!Genome.isGenomeFeasible(finalRandomNodes)) {
                            continue;
                        }

                        const finalRandomBudget = validateBudget(randomGenome);
                        if (!finalRandomBudget.ok) {
                            continue;
                        }

                        genomes.push({
                            id: crypto.randomUUID(),
                            genome: randomGenome,
                            nodes: finalRandomNodes
                        });
                    } catch (e) {
                        console.warn(`[initPopulation] Random generation failed. Attempt ${randomAttempts}/${numRandom * 12}`, e);
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

                const seedBudget = validateBudget(genome);
                if (!seedBudget.ok) {
                    addLog(`Seed genome rejected by memory budget: ${seedBudget.reason}`, "warn");
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
            while (genomes.length < popSize && initAttempts < popSize * INIT_ATTEMPT_MULTIPLIER) {
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
                    const maxMutAttempts = STRUCTURAL_RETRY_ATTEMPTS;

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
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                clone = res.genome;
                                mutated = true;
                            }
                        }
                    }

                    if (!mutated) {
                        // Structural mutation couldn't produce a valid alternative; retry from another seed/clone.
                        continue;
                    }

                    // Additional probabilistic mutation rounds for extra variance
                    const extraRounds = Math.floor(Math.random() * 3);
                    for (let r = 0; r < extraRounds; r++) {
                        const dynamicRates = settings.useAdaptiveMutation
                            ? getAdaptiveMutationRates(settings, clone.getAllNodes().length)
                            : {
                                addNode: settings.mutationRates.addNode,
                                removeNode: settings.mutationRates.removeNode,
                                removeSubgraph: settings.mutationRates.removeSubgraph
                            };

                        if (Math.random() < dynamicRates.removeSubgraph) {
                            const res = clone.MutateRemoveSubgraph();
                            if (res) {
                                const candidateBudget = validateBudget(res.genome);
                                if (candidateBudget.ok) clone = res.genome;
                            }
                        }
                        if (Math.random() < dynamicRates.removeNode) {
                            const res = clone.MutateRemoveNode();
                            if (res) {
                                const candidateBudget = validateBudget(res.genome);
                                if (candidateBudget.ok) clone = res.genome;
                            }
                        }
                        if (Math.random() < dynamicRates.addNode) {
                            const res = clone.MutateAddNode(maxNodes);
                            if (res) {
                                const candidateBudget = validateBudget(res.genome);
                                if (candidateBudget.ok) clone = res.genome;
                            }
                        }
                        if (settings.mutationRates.addSkipConnection && Math.random() < settings.mutationRates.addSkipConnection) {
                            const res = clone.MutateAddSkipConnection(maxNodes);
                            if (res) {
                                const candidateBudget = validateBudget(res.genome);
                                if (candidateBudget.ok) clone = res.genome;
                            }
                        }
                        if (settings.mutationRates.changeLayerType && Math.random() < settings.mutationRates.changeLayerType) {
                            const res = clone.MutateChangeLayerType(maxNodes);
                            if (res) {
                                const candidateBudget = validateBudget(res.genome);
                                if (candidateBudget.ok) clone = res.genome;
                            }
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

                    const finalBudget = validateBudget(clone);
                    if (!finalBudget.ok) {
                        if (initAttempts % 10 === 0) {
                            addLog(`Retrying mutation (memory budget exceeded)...`, "warn");
                        }
                        continue;
                    }

                    genomes.push({
                        id: crypto.randomUUID(),
                        genome: clone,
                        nodes: finalNodes
                    });
                } catch (e) {
                    console.warn(
                        `[initPopulation] Seed mutation failed (likely shape mismatch). Attempt ${initAttempts}/${popSize * INIT_ATTEMPT_MULTIPLIER}`,
                        e,
                    );
                    continue;
                }
            }

            if (genomes.length < popSize && genomes.length > 0) {
                addLog(
                    `Initialization recovery: population incomplete (${genomes.length}/${popSize}), retrying with additional mutation passes...`,
                    'warn',
                );

                let recoveryAttempts = 0;
                while (genomes.length < popSize && recoveryAttempts < popSize * INIT_ATTEMPT_MULTIPLIER) {
                    recoveryAttempts++;

                    try {
                        const base = genomes[Math.floor(Math.random() * genomes.length)]?.genome;
                        if (!base) {
                            break;
                        }

                        const baseSerialized = await serializeGenome(base);
                        const { genome: candidate } = await deserializeGenome(baseSerialized);
                        const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

                        let accepted: Genome | null = null;
                        for (let m = 0; m < STRUCTURAL_RETRY_ATTEMPTS && !accepted; m++) {
                            const options = [
                                () => candidate.MutateAddNode(maxNodes),
                                () => candidate.MutateRemoveNode(),
                                () => candidate.MutateRemoveSubgraph(),
                                ...(settings.mutationRates.addSkipConnection ? [() => candidate.MutateAddSkipConnection(maxNodes)] : []),
                                ...(settings.mutationRates.changeLayerType ? [() => candidate.MutateChangeLayerType(maxNodes)] : []),
                            ];

                            const picked = options[Math.floor(Math.random() * options.length)];
                            const result = picked();
                            if (!result) {
                                continue;
                            }

                            const nodes = result.genome.getAllNodes();
                            if (!Genome.isGenomeFeasible(nodes)) {
                                continue;
                            }
                            const mem = validateBudget(result.genome);
                            if (!mem.ok) {
                                continue;
                            }
                            accepted = result.genome;
                        }

                        if (!accepted) {
                            continue;
                        }

                        genomes.push({
                            id: crypto.randomUUID(),
                            genome: accepted,
                            nodes: accepted.getAllNodes(),
                        });
                    } catch {
                        continue;
                    }
                }
            }

            // If we have more seeds than popSize, truncate
            if (genomes.length > popSize) {
                genomes.length = popSize;
            }

            genealogyMapRef.current = new Map();
            if (settings.genealogyTrackingEnabled) {
                for (const genome of genomes) {
                    await registerGenealogyEvent(
                        genome.id,
                        0,
                        [],
                        { type: 'Random' },
                        { source: 'generation_0' },
                    );
                }
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
                evaluated: false,
                genealogy: settings.genealogyTrackingEnabled ? new Map(genealogyMapRef.current) : undefined,
            }]);

            addLog(`Spawned Generation 0: ${numFromSeeds} direct seeds, ${numRandom} random archs, ${Math.max(0, genomes.length - numFromSeeds - numRandom)} mutated clones.`, "info");
        } catch (e) {
            addLog(`Failed to initialize population: ${String(e)}`, "error");
        }
    }, [addLog, settings, datasetProfileId, registerGenealogyEvent, getGenomeBudgetValidator]);

    // Main Async Loop
    const runGeneration = useCallback(async () => {
        if (!isRunningRef.current || population.length === 0 || !datasetProfileId) return;
        if (generationRunInFlightRef.current) {
            return;
        }

        generationRunInFlightRef.current = true;

        try {

        addLog(`--- Starting Generation ${generation} Evaluation ---`, "info");

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
                    if (!isRunningRef.current || isPausedRef.current) {
                        addLog('Zero-cost scoring interrupted by stop/pause request.', 'warn');
                        break;
                    }

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
                        const errMsg = String(e);
                        if (/cancel|stopp|aborted/i.test(errMsg)) {
                            addLog('Zero-cost scoring cancelled.', 'warn');
                            break;
                        }

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
            const currentProfile = datasetProfiles.find(p => p.id === datasetProfileId);
            const trainSplit = currentProfile?.split?.train ?? 80;
            const valSplit = currentProfile?.split?.val ?? 10;
            const testSplit = currentProfile?.split?.test ?? 10;

            // 2. Call Rust Evaluator
            // For now, we use a global evalEpochs value
            // In a more advanced implementation, we'd modify evaluate_population to accept per-genome epochs
            const perGenomeEpochs = Array.from({ length: serializedGenomes.length }, (_, i) => 
                evalEpochsAdjustments.get(i) ?? settings.evalEpochs
            );

            const configuredRamMb = settings.customDeviceParams?.ram_mb
                ?? Math.max(1, settings.resourceTargets.ram / MB_TO_BYTES);
            const safetyMarginMb = Math.max(0, settings.memorySafetyMarginMb ?? 128);
            const usableRamMb = Math.max(1, configuredRamMb - safetyMarginMb);
            const perGenomeEstimateMb = Math.max(64, Math.floor(safetyMarginMb * 0.75));
            const memoryFitParallelJobs = Math.max(1, Math.floor(usableRamMb / perGenomeEstimateMb));

            const configuredParallelJobs = Math.max(1, settings.maxParallelJobs ?? 1);
            const explicitParallelOverride = settings.executionMode !== 'sequential' && configuredParallelJobs > 1;
            const effectiveTargetParallelJobs = settings.executionMode === 'sequential'
                ? 1
                : (configuredParallelJobs <= 1 ? memoryFitParallelJobs : configuredParallelJobs);

            // If the user explicitly sets parallel jobs > 1, trust that value.
            // The memory-fit estimate is still useful as an auto/default picker, but not as a hard cap.
            const effectiveParallelCap = explicitParallelOverride
                ? effectiveTargetParallelJobs
                : Math.min(effectiveTargetParallelJobs, memoryFitParallelJobs);

            const requestedMaxParallelJobs = Math.max(
                1,
                Math.min(
                    effectiveParallelCap,
                    serializedGenomes.length,
                ),
            );
            const requestedExecutionMode = settings.executionMode
                ?? (requestedMaxParallelJobs > 1 ? 'parallel-safe-limited' : 'sequential');
            
            const results = await invoke<EvaluationResult[]>('evaluate_population', {
                genomes: serializedGenomes,
                datasetProfile: datasetProfileId,
                batchSize: settings.batchSize || 32,
                perGenomeEpochs,
                datasetPercent: settings.datasetPercent || 100,
                trainSplit,
                valSplit,
                testSplit,
                genomeIds: population.map((p) => p.id),
                sourceGeneration: generation,
                maxParallelJobs: requestedMaxParallelJobs,
                executionMode: requestedExecutionMode,
                memorySafetyMarginMb: safetyMarginMb,
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
                    profiler: res.profiler,
                } as PopulatedGenome;
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
            const totalTrainingMs = evaluatedPop.reduce(
                (acc, p) => acc + (p.profiler?.total_train_duration_ms ?? 0),
                0,
            );
            const totalInferenceMs = evaluatedPop.reduce(
                (acc, p) => acc + (p.profiler?.inference_msec_per_sample ?? 0),
                0,
            );
            const avgSamplesPerSec = evaluatedPop.length > 0
                ? evaluatedPop.reduce((acc, p) => acc + (p.profiler?.samples_per_sec ?? 0), 0) / evaluatedPop.length
                : 0;

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
                    evaluated: true,
                    totalTrainingMs,
                    totalInferenceMs,
                    avgSamplesPerSec,
                };
                return updated;
            });

            // Clear per-genome metrics for next generation
            perGenomeMetricsRef.current = new Map();

            // 5. Check if we should stop
            if (!isRunningRef.current) return;

            if (isPausedRef.current) {
                addLog(`Paused after generation ${generation}.`, 'warn');
                return;
            }

            // Max generations auto-stop
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

            // Simple Tournament Selection
            const tournamentSelect = () => {
                const i1 = Math.floor(Math.random() * popSize);
                const i2 = Math.floor(Math.random() * popSize);
                return (evaluatedPop[i1].adjustedFitness || 0) > (evaluatedPop[i2].adjustedFitness || 0) ? evaluatedPop[i1] : evaluatedPop[i2];
            };

            const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;
            const validateBudget = getGenomeBudgetValidator();

            const nextGenNum = generation + 1;
            let breedAttempts = 0;
            while (nextGen.length < popSize && breedAttempts < popSize * BREED_ATTEMPT_MULTIPLIER) {
                breedAttempts++;
                try {
                    const parentAEntry = tournamentSelect()!;
                    const parentBEntry = tournamentSelect()!;
                    const parentA = parentAEntry.genome;
                    const parentB = parentBEntry.genome;
                    let childParentIds: string[] = [parentAEntry.id];
                    let childMutationType: MutationType = { type: 'Random' };
                    const childMutationParams: Record<string, unknown> = {};

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
                                if (res) {
                                    childGenome = res.genome;
                                    childParentIds = [parentAEntry.id, parentBEntry.id];
                                    childMutationType = {
                                        type: 'Crossover',
                                        data: { parent1: parentAEntry.id, parent2: parentBEntry.id },
                                    };
                                    childMutationParams.crossover_strategy = strategy;
                                }
                            } else if (strategy === 'neat-style') {
                                const res = parentA.BreedNeatStyle(parentB, maxNodes);
                                if (res) {
                                    childGenome = res.genome;
                                    childParentIds = [parentAEntry.id, parentBEntry.id];
                                    childMutationType = {
                                        type: 'Crossover',
                                        data: { parent1: parentAEntry.id, parent2: parentBEntry.id },
                                    };
                                    childMutationParams.crossover_strategy = strategy;
                                }
                            } else if (strategy === 'multi-point') {
                                const res = parentA.BreedMultiPoint(parentB, maxNodes);
                                if (res) {
                                    childGenome = res.genome;
                                    childParentIds = [parentAEntry.id, parentBEntry.id];
                                    childMutationType = {
                                        type: 'Crossover',
                                        data: { parent1: parentAEntry.id, parent2: parentBEntry.id },
                                    };
                                    childMutationParams.crossover_strategy = strategy;
                                }
                            } else {
                                // Default 'subgraph-insertion'
                                const res = parentA.Breed(parentB, maxNodes);
                                if (res) {
                                    childGenome = res.genome;
                                    childParentIds = [parentAEntry.id, parentBEntry.id];
                                    childMutationType = {
                                        type: 'Crossover',
                                        data: { parent1: parentAEntry.id, parent2: parentBEntry.id },
                                    };
                                    childMutationParams.crossover_strategy = strategy;
                                }
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
                        childParentIds = [parentAEntry.id];
                        childMutationType = { type: 'Random' };
                        childMutationParams.crossover_fallback = true;
                    }

                    // Mutation
                    const dynamicRates = settings.useAdaptiveMutation
                        ? getAdaptiveMutationRates(settings, childGenome.getAllNodes().length)
                        : {
                            addNode: settings.mutationRates.addNode,
                            removeNode: settings.mutationRates.removeNode,
                            removeSubgraph: settings.mutationRates.removeSubgraph
                        };

                    if (Math.random() < dynamicRates.removeSubgraph) {
                        const res = childGenome.MutateRemoveSubgraph();
                        if (res) {
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                childGenome = res.genome;
                                childMutationType = { type: 'RemoveSubgraph', data: { node_ids: [] } };
                                childMutationParams.last_structural_mutation = 'remove_subgraph';
                            }
                        }
                    }
                    if (Math.random() < dynamicRates.removeNode) {
                        const res = childGenome.MutateRemoveNode();
                        if (res) {
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                childGenome = res.genome;
                                childMutationType = { type: 'RemoveNode', data: { node_id: '' } };
                                childMutationParams.last_structural_mutation = 'remove_node';
                            }
                        }
                    }
                    if (Math.random() < dynamicRates.addNode) {
                        const res = childGenome.MutateAddNode(maxNodes);
                        if (res) {
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                childGenome = res.genome;
                                childMutationType = {
                                    type: 'AddNode',
                                    data: { node_type: 'unknown', source: 'unknown', target: 'unknown' },
                                };
                                childMutationParams.last_structural_mutation = 'add_node';
                            }
                        }
                    }
                    if (settings.mutationRates.addSkipConnection && Math.random() < settings.mutationRates.addSkipConnection) {
                        const res = childGenome.MutateAddSkipConnection(maxNodes);
                        if (res) {
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                childGenome = res.genome;
                                childMutationType = {
                                    type: 'ParameterMutation',
                                    data: { layer_id: 'graph', param_name: 'add_skip_connection' },
                                };
                                childMutationParams.last_structural_mutation = 'add_skip_connection';
                            }
                        }
                    }
                    if (settings.mutationRates.changeLayerType && Math.random() < settings.mutationRates.changeLayerType) {
                        const res = childGenome.MutateChangeLayerType(maxNodes);
                        if (res) {
                            const candidateBudget = validateBudget(res.genome);
                            if (candidateBudget.ok) {
                                childGenome = res.genome;
                                childMutationType = {
                                    type: 'ParameterMutation',
                                    data: { layer_id: 'graph', param_name: 'change_layer_type' },
                                };
                                childMutationParams.last_structural_mutation = 'change_layer_type';
                            }
                        }
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
                        childMutationType = {
                            type: 'ParameterMutation',
                            data: { layer_id: 'multiple', param_name: 'params' },
                        };
                        childMutationParams.params_mutation_rate = settings.mutationRates.params;
                    }

                    const nextGenNodes = childGenome.getAllNodes();
                    if (!Genome.isGenomeFeasible(nextGenNodes)) {
                        if (breedAttempts % 10 === 0) {
                            addLog(`Rejected invalid child (shape mismatch). Retrying breeding...`, "warn");
                        }
                        continue;
                    }

                    const childBudget = validateBudget(childGenome);
                    if (!childBudget.ok) {
                        if (breedAttempts % 10 === 0) {
                            addLog(`Rejected child by memory budget: ${childBudget.reason}. Retrying breeding...`, "warn");
                        }
                        continue;
                    }

                    const childId = crypto.randomUUID();
                    await registerGenealogyEvent(
                        childId,
                        nextGenNum,
                        childParentIds,
                        childMutationType,
                        childMutationParams,
                    );

                    nextGen.push({
                        id: childId,
                        genome: childGenome,
                        nodes: nextGenNodes,
                        generation: nextGenNum,
                        parent_ids: childParentIds,
                        mutation_type: childMutationType,
                        mutation_params: childMutationParams,
                    });
                } catch (e) {
                    console.warn(
                        `[Breed] Child generation failed (likely shape mismatch). Attempt ${breedAttempts}/${popSize * BREED_ATTEMPT_MULTIPLIER}`,
                        e,
                    );
                    continue;
                }
            }

            if (nextGen.length < popSize && evaluatedPop.length > 0) {
                addLog(
                    `Breeding recovery: generated ${nextGen.length}/${popSize}. Retrying with forced alternative mutations...`,
                    'warn',
                );

                let recoveryAttempts = 0;
                while (nextGen.length < popSize && recoveryAttempts < popSize * BREED_ATTEMPT_MULTIPLIER) {
                    recoveryAttempts++;
                    try {
                        const parentEntry = tournamentSelect();
                        const parentSerialized = await serializeGenome(parentEntry.genome);
                        const { genome: candidate } = await deserializeGenome(parentSerialized);

                        let acceptedGenome: Genome | null = null;
                        let mutationLabel = 'recovery_mutation';
                        for (let m = 0; m < STRUCTURAL_RETRY_ATTEMPTS && !acceptedGenome; m++) {
                            const options = [
                                () => candidate.MutateAddNode(maxNodes),
                                () => candidate.MutateRemoveNode(),
                                () => candidate.MutateRemoveSubgraph(),
                                ...(settings.mutationRates.addSkipConnection ? [() => candidate.MutateAddSkipConnection(maxNodes)] : []),
                                ...(settings.mutationRates.changeLayerType ? [() => candidate.MutateChangeLayerType(maxNodes)] : []),
                            ];

                            const pickedIndex = Math.floor(Math.random() * options.length);
                            const result = options[pickedIndex]();
                            if (!result) {
                                continue;
                            }

                            const nodes = result.genome.getAllNodes();
                            if (!Genome.isGenomeFeasible(nodes)) {
                                continue;
                            }
                            const mem = validateBudget(result.genome);
                            if (!mem.ok) {
                                continue;
                            }

                            acceptedGenome = result.genome;
                            mutationLabel = ['add_node', 'remove_node', 'remove_subgraph', 'add_skip_connection', 'change_layer_type'][pickedIndex] || 'recovery_mutation';
                        }

                        if (!acceptedGenome) {
                            continue;
                        }

                        const childId = crypto.randomUUID();
                        const mutationType: MutationType = {
                            type: 'ParameterMutation',
                            data: { layer_id: 'graph', param_name: mutationLabel },
                        };
                        const mutationParams: Record<string, unknown> = { recovery: true, last_structural_mutation: mutationLabel };

                        await registerGenealogyEvent(
                            childId,
                            nextGenNum,
                            [parentEntry.id],
                            mutationType,
                            mutationParams,
                        );

                        nextGen.push({
                            id: childId,
                            genome: acceptedGenome,
                            nodes: acceptedGenome.getAllNodes(),
                            generation: nextGenNum,
                            parent_ids: [parentEntry.id],
                            mutation_type: mutationType,
                            mutation_params: mutationParams,
                        });
                    } catch {
                        continue;
                    }
                }
            }

            if (nextGen.length < popSize && nextGen.length > 0) {
                addLog(
                    `Unable to produce enough distinct valid children under current constraints (${nextGen.length}/${popSize}). Filling the remainder with elite clones.`,
                    'warn',
                );

                let fallbackIndex = 0;
                while (nextGen.length < popSize) {
                    const fallbackParent = evaluatedPop[fallbackIndex % Math.max(1, eliteCount)];
                    fallbackIndex++;

                    const fallbackSerialized = await serializeGenome(fallbackParent.genome);
                    const { genome: fallbackClone } = await deserializeGenome(fallbackSerialized);
                    const fallbackNodes = fallbackClone.getAllNodes();

                    if (!Genome.isGenomeFeasible(fallbackNodes)) {
                        continue;
                    }
                    const fallbackBudget = validateBudget(fallbackClone);
                    if (!fallbackBudget.ok) {
                        continue;
                    }

                    const fallbackId = crypto.randomUUID();
                    await registerGenealogyEvent(
                        fallbackId,
                        nextGenNum,
                        [fallbackParent.id],
                        { type: 'Random' },
                        { fallback_fill: true },
                    );

                    nextGen.push({
                        id: fallbackId,
                        genome: fallbackClone,
                        nodes: fallbackNodes,
                        generation: nextGenNum,
                        parent_ids: [fallbackParent.id],
                        mutation_type: { type: 'Random' },
                        mutation_params: { fallback_fill: true },
                    });
                }
            }

            setPopulation(nextGen);
            setGeneration(g => g + 1);

            // Save pre-eval snapshot for the new generation immediately
            setGenerationHistory(prev => [...prev, {
                generation: nextGenNum,
                genomes: [...nextGen],
                bestFitness: 0,
                avgNodes: Math.round(nextGen.reduce((acc, g) => acc + g.nodes.length, 0) / nextGen.length),
                timestamp: new Date().toLocaleTimeString(),
                evaluated: false,
                genealogy: settings.genealogyTrackingEnabled ? new Map(genealogyMapRef.current) : undefined,
            }]);

            // Timeout to yield to React rendering before next loop iteration
            setTimeout(() => {
                if (isRunningRef.current) {
                    runGeneration();
                }
            }, 100);

        } catch (err) {
            console.error('evaluate_population error:', err);
            const msg = String(err);
            addLog(`Evaluation failed: ${msg}`, "error");

            if (
                msg.includes('No training batches could be assembled') ||
                msg.includes('ipc protocol failed') ||
                msg.includes('Failed to fetch') ||
                msg.includes('panicked during training/validation') ||
                msg.includes('corrupted WGPU state') ||
                msg.includes('Another evaluation is already running')
            ) {
                addLog(
                    'Evolution stopped: backend runtime entered an invalid state. Stop and restart Tauri app before the next run.',
                    'warn',
                );
                stopEvolution();
            }
        } finally {
            generationRunInFlightRef.current = false;
        }

    }, [datasetProfileId, generation, population, settings, addLog, stopEvolution, getGenomeBudgetValidator]);

    const startEvolution = useCallback((seedGenomes: string[]) => {
        try {
            genealogyMapRef.current = new Map();
            if (!datasetProfileId) {
                addLog("Cannot start: No dataset selected!", "error");
                return;
            }

            if (seedGenomes.length === 0 && !settings.useRandomInitialization) {
                addLog("Cannot start: No seeds provided and Random Initialization disabled!", "error");
                return;
            }

            // === PHASE 3: Validate dataset profile before evolution ===
            const profile = datasetProfiles.find(p => p.id === datasetProfileId);
            
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
            setIsPaused(false);
            isRunningRef.current = true;
            isPausedRef.current = false;

            initPopulation(seedGenomes);
        } catch (err: any) {
            console.error("startEvolution error:", err);
            addLog(`Error during startEvolution: ${err.message || String(err)}`, "error");
        }
    }, [datasetProfileId, addLog, initPopulation, settings]);

    useEffect(() => {
        if (isRunning && !isPaused && population.length > 0) {
            // Need a slight delay to allow React rendering/logging to flush
            const timer = setTimeout(() => {
                runGeneration();
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [isRunning, isPaused, population, runGeneration]);

    return {
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
        stats,
        runGeneration, // Exposed to be called manually or via useEffect
        currentEvaluatingIndex,
        liveMetrics,
        generationHistory
    };
};
