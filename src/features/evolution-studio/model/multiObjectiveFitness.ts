/**
 * Multi-Objective Fitness Calculator
 * 
 * Provides vector-based fitness evaluation for Pareto optimization.
 * All metrics are computed WITHOUT training (except accuracy).
 * 
 * @see useEvolutionLoop.ts - integration point
 */

/**
 * Objective vector for multi-objective optimization
 * 
 * Quality: maximize (higher = better)
 * Resources: minimize (lower = better)
 */
export interface ObjectiveVector {
    // Quality metric
    quality: number;       // accuracy (if trained) OR proxy_score (if skipped)
    
    // Resource consumption (KB)
    flashKB: number;       // Flash memory (parameters)
    ramKB: number;         // RAM (active memory during inference)
    
    // Performance metrics
    macs: number;          // Millions of MACs (Multiply-Accumulate operations)
    estimatedLatencyMs: number;  // Estimated inference latency
}

/**
 * Constraint violation for resource limits
 */
export interface ConstraintViolation {
    type: 'flash' | 'ram' | 'macs';
    limit: number;
    actual: number;
    violation: number;  // How much over the limit
}

/**
 * Multi-objective fitness result
 */
export interface MultiObjectiveResult {
    vector: ObjectiveVector;
    paretoRank: number;
    crowdingDistance: number;
    isFeasible: boolean;   // true if satisfies all resource constraints
    violations: ConstraintViolation[];
}

/**
 * Creates objective vector from evaluation results
 * 
 * @param accuracy - Model accuracy (undefined if not trained)
 * @param proxyScore - Zero-cost proxy score (undefined if not computed)
 * @param resources - Resource consumption from genome
 * @param wasSkipped - true if zero-cost skip was applied (no training)
 * @returns Objective vector for Pareto optimization
 */
export function createObjectiveVector(
    accuracy: number | undefined,
    proxyScore: number | undefined,
    resources: { totalFlash: number; totalRam: number; totalMacs: number },
    wasSkipped: boolean
): ObjectiveVector {
    // Quality: use proxy if skipped, otherwise accuracy
    const quality = wasSkipped 
        ? (proxyScore || 0) 
        : (accuracy !== undefined ? accuracy : 0);
    
    return {
        quality,
        flashKB: resources.totalFlash / 1024,
        ramKB: resources.totalRam / 1024,
        macs: resources.totalMacs / 1_000_000,
        estimatedLatencyMs: estimateLatencyFromMACs(resources.totalMacs)
    };
}

/**
 * Check if objective vector satisfies resource constraints
 * 
 * @param vector - Objective vector to check
 * @param constraints - Resource limits (undefined = no limit)
 * @returns Feasibility status and violations
 */
export function checkConstraints(
    vector: ObjectiveVector,
    constraints: {
        maxFlashKB?: number;
        maxRamKB?: number;
        maxMacs?: number;
    }
): { isFeasible: boolean; violations: ConstraintViolation[] } {
    const violations: ConstraintViolation[] = [];
    
    if (constraints.maxFlashKB && vector.flashKB > constraints.maxFlashKB) {
        violations.push({
            type: 'flash',
            limit: constraints.maxFlashKB,
            actual: vector.flashKB,
            violation: vector.flashKB - constraints.maxFlashKB
        });
    }
    
    if (constraints.maxRamKB && vector.ramKB > constraints.maxRamKB) {
        violations.push({
            type: 'ram',
            limit: constraints.maxRamKB,
            actual: vector.ramKB,
            violation: vector.ramKB - constraints.maxRamKB
        });
    }
    
    if (constraints.maxMacs && vector.macs > constraints.maxMacs) {
        violations.push({
            type: 'macs',
            limit: constraints.maxMacs,
            actual: vector.macs,
            violation: vector.macs - constraints.maxMacs
        });
    }
    
    return {
        isFeasible: violations.length === 0,
        violations
    };
}

/**
 * Estimate inference latency from MACs count
 * 
 * Simplified model: 1M MACs ≈ 1ms on Edge TPU
 * Can be calibrated for specific target hardware
 * 
 * @param macs - Total MACs count
 * @returns Estimated latency in milliseconds
 */
function estimateLatencyFromMACs(macs: number): number {
    // Base latency: 1ms per 1M MACs (Edge TPU reference)
    const baseLatency = macs / 1_000_000;
    
    // Add overhead for memory access (simplified model)
    const memoryOverhead = 0.1; // 10% overhead
    
    return baseLatency * (1 + memoryOverhead);
}

/**
 * Normalize objective value to [0, 1] range
 * Used for hypervolume calculation
 * 
 * @param value - Raw objective value
 * @param min - Minimum expected value
 * @param max - Maximum expected value
 * @param maximize - true if higher is better (quality), false if lower is better (resources)
 * @returns Normalized value in [0, 1]
 */
export function normalizeObjective(
    value: number,
    min: number,
    max: number,
    maximize: boolean
): number {
    const range = max - min;
    if (range === 0) return 0.5; // All values are the same
    
    const normalized = (value - min) / range;
    return maximize ? normalized : (1 - normalized);
}

/**
 * Calculate scalar fitness from objective vector (for single-objective mode)
 * 
 * This maintains backward compatibility with existing single-objective evolution
 * 
 * @param vector - Objective vector
 * @param settings - Fitness calculation settings
 * @returns Scalar fitness value
 */
export function calculateScalarFitness(
    vector: ObjectiveVector,
    options: {
        useParsimonyPressure: boolean;
        parsimonyAlpha: number;
        nodeCount: number;
        useResourceAwareFitness: boolean;
        resourceTargets: { flash: number; ram: number; macs: number };
    }
): number {
    // Base fitness = quality (accuracy or proxy)
    let baseFitness = vector.quality;
    
    // Resource-aware fitness penalty
    if (options.useResourceAwareFitness) {
        const flashPenalty = Math.max(0, vector.flashKB * 1024 - options.resourceTargets.flash) / options.resourceTargets.flash;
        const ramPenalty = Math.max(0, vector.ramKB * 1024 - options.resourceTargets.ram) / options.resourceTargets.ram;
        const macsPenalty = Math.max(0, vector.macs * 1_000_000 - options.resourceTargets.macs) / options.resourceTargets.macs;
        const resourcePenalty = (flashPenalty + ramPenalty + macsPenalty) / 3;
        baseFitness -= resourcePenalty;
    }
    
    // Parsimony pressure (penalize large networks)
    if (options.useParsimonyPressure) {
        baseFitness -= options.parsimonyAlpha * options.nodeCount;
    }
    
    return baseFitness;
}
