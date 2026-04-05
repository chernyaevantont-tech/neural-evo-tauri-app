/**
 * Pareto Sorting and Ranking (NSGA-II style)
 * 
 * Implements non-dominated sorting and crowding distance calculation
 * for multi-objective evolutionary optimization.
 * 
 * @see multiObjectiveFitness.ts - ObjectiveVector type
 */

import { ObjectiveVector } from './multiObjectiveFitness';

/**
 * Pareto ranking result for a single genome
 */
export interface ParetoRanking {
    rank: number;           // 0 = Pareto front (non-dominated)
    crowdingDistance: number;  // Diversity metric (higher = more diverse)
    dominatedCount: number; // Number of solutions this one dominates
}

/**
 * Population item for Pareto sorting
 */
export interface ParetoPopulationItem {
    id: string;
    vector: ObjectiveVector;
}

/**
 * Non-dominated sorting (NSGA-II algorithm)
 * 
 * Assigns Pareto ranks to all individuals:
 * - Rank 0: Non-dominated solutions (Pareto front)
 * - Rank 1: Solutions dominated only by rank 0
 * - Rank 2: Solutions dominated only by rank 0 and 1
 * - etc.
 * 
 * @param population - Population to sort
 * @returns Map of genome ID to Pareto ranking
 */
export function nonDominatedSorting(
    population: ParetoPopulationItem[]
): Map<string, ParetoRanking> {
    const n = population.length;
    const rankings = new Map<string, ParetoRanking>();
    
    // Initialize rankings
    for (const p of population) {
        rankings.set(p.id, {
            rank: 0,
            crowdingDistance: 0,
            dominatedCount: 0
        });
    }
    
    // Build domination sets
    const fronts: string[][] = [[]]; // Fronts: front[0] = rank 0, etc.
    const dominatedSets = new Map<string, Set<string>>();
    
    // For each pair, determine domination
    for (let i = 0; i < n; i++) {
        const domSet = new Set<string>();
        let dominatedCount = 0;
        
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            
            if (dominates(population[i].vector, population[j].vector)) {
                domSet.add(population[j].id);
                dominatedCount++;
            }
        }
        
        dominatedSets.set(population[i].id, domSet);
        rankings.get(population[i].id)!.dominatedCount = dominatedCount;
        
        // If not dominated by anyone, assign to front 0
        if (dominatedCount === 0) {
            rankings.get(population[i].id)!.rank = 0;
            fronts[0].push(population[i].id);
        }
    }
    
    // Build remaining fronts iteratively
    let currentFrontIdx = 0;
    
    while (fronts[currentFrontIdx].length > 0) {
        const nextFront: string[] = [];
        const currentFront = fronts[currentFrontIdx];
        
        for (const id of currentFront) {
            const domSet = dominatedSets.get(id)!;
            
            for (const dominatedId of domSet) {
                const ranking = rankings.get(dominatedId)!;
                ranking.dominatedCount--;
                
                // If all dominators are in previous fronts, this goes to next front
                if (ranking.dominatedCount === 0) {
                    ranking.rank = currentFrontIdx + 1;
                    nextFront.push(dominatedId);
                }
            }
        }
        
        fronts.push(nextFront);
        currentFrontIdx++;
    }
    
    // Calculate crowding distance for diversity
    calculateCrowdingDistance(population, rankings);
    
    return rankings;
}

/**
 * Check if vector A dominates vector B
 * 
 * A dominates B if:
 * - A is not worse than B in ALL objectives
 * - A is strictly better than B in AT LEAST ONE objective
 * 
 * Objectives:
 * - quality: maximize (higher = better)
 * - flashKB, ramKB, macs, latency: minimize (lower = better)
 * 
 * @param a - Vector A
 * @param b - Vector B
 * @returns true if A dominates B
 */
export function dominates(a: ObjectiveVector, b: ObjectiveVector): boolean {
    // Check if A is not worse than B in all objectives
    const notWorse = 
        a.quality >= b.quality &&           // Higher quality is better
        a.flashKB <= b.flashKB &&           // Lower flash is better
        a.ramKB <= b.ramKB &&               // Lower RAM is better
        a.macs <= b.macs;                   // Lower MACs is better
    
    // Check if A is strictly better than B in at least one objective
    const better = 
        a.quality > b.quality ||
        a.flashKB < b.flashKB ||
        a.ramKB < b.ramKB ||
        a.macs < b.macs;
    
    return notWorse && better;
}

/**
 * Calculate crowding distance for all individuals
 * 
 * Crowding distance measures how isolated a solution is in objective space.
 * Higher distance = more isolated = more diverse = preferred for selection.
 * 
 * Boundary solutions get Infinity distance (always preserved).
 * 
 * @param population - Population with rankings
 * @param rankings - Pareto rankings map (modified in place)
 */
export function calculateCrowdingDistance(
    population: ParetoPopulationItem[],
    rankings: Map<string, ParetoRanking>
): void {
    // Group by front
    const fronts = new Map<number, string[]>();
    
    for (const [id, ranking] of rankings) {
        if (!fronts.has(ranking.rank)) {
            fronts.set(ranking.rank, []);
        }
        fronts.get(ranking.rank)!.push(id);
    }
    
    // Calculate distance for each front
    for (const [_rank, front] of fronts) {
        if (front.length <= 2) {
            // All solutions in small front are boundary points
            for (const id of front) {
                rankings.get(id)!.crowdingDistance = Infinity;
            }
            continue;
        }
        
        // Objectives to consider (excluding latency which is derived from MACs)
        const objectives: (keyof ObjectiveVector)[] = ['quality', 'flashKB', 'ramKB', 'macs'];
        
        for (const obj of objectives) {
            // Sort front by this objective
            const sorted = [...front].sort((a, b) => {
                const vecA = population.find(p => p.id === a)!.vector;
                const vecB = population.find(p => p.id === b)!.vector;
                return (vecA[obj] as number) - (vecB[obj] as number);
            });
            
            // Boundary solutions get infinity
            rankings.get(sorted[0])!.crowdingDistance += Infinity;
            rankings.get(sorted[sorted.length - 1])!.crowdingDistance += Infinity;
            
            // Calculate range for normalization
            const minVec = population.find(p => p.id === sorted[0])!.vector;
            const maxVec = population.find(p => p.id === sorted[sorted.length - 1])!.vector;
            const range = (maxVec[obj] as number) - (minVec[obj] as number);
            
            if (range === 0) continue; // All values are the same
            
            // Add normalized distance for intermediate solutions
            for (let i = 1; i < sorted.length - 1; i++) {
                const id = sorted[i];
                const prevVec = population.find(p => p.id === sorted[i - 1])!.vector;
                const nextVec = population.find(p => p.id === sorted[i + 1])!.vector;
                
                const distance = ((nextVec[obj] as number) - (prevVec[obj] as number)) / range;
                rankings.get(id)!.crowdingDistance += distance;
            }
        }
    }
}

/**
 * Select better individual based on Pareto ranking
 * 
 * Comparison rules:
 * 1. Lower rank is better (closer to Pareto front)
 * 2. If same rank, higher crowding distance is better (more diverse)
 * 
 * @param a - Individual A with ranking
 * @param b - Individual B with ranking
 * @returns 'a' if A is better, 'b' if B is better
 */
export function comparePareto(
    a: { id: string; ranking: ParetoRanking },
    b: { id: string; ranking: ParetoRanking }
): 'a' | 'b' {
    // Compare ranks (lower is better)
    if (a.ranking.rank !== b.ranking.rank) {
        return a.ranking.rank < b.ranking.rank ? 'a' : 'b';
    }
    
    // Same rank: compare crowding distance (higher is better)
    if (a.ranking.crowdingDistance !== b.ranking.crowdingDistance) {
        return a.ranking.crowdingDistance > b.ranking.crowdingDistance ? 'a' : 'b';
    }
    
    // Equal: prefer A (arbitrary but deterministic)
    return 'a';
}

/**
 * Get Pareto front (rank 0) from population
 * 
 * @param population - Population with rankings
 * @param rankings - Pareto rankings
 * @returns Array of IDs in the Pareto front
 */
export function getParetoFront(
    population: ParetoPopulationItem[],
    rankings: Map<string, ParetoRanking>
): ParetoPopulationItem[] {
    return population.filter(p => rankings.get(p.id)!.rank === 0);
}

/**
 * Sort population by Pareto ranking (for display/selection)
 * 
 * Sort order:
 * 1. Rank (ascending - front 0 first)
 * 2. Crowding distance (descending - more diverse first)
 * 
 * @param population - Population with rankings
 * @param rankings - Pareto rankings
 * @returns Sorted population
 */
export function sortPareto(
    population: ParetoPopulationItem[],
    rankings: Map<string, ParetoRanking>
): ParetoPopulationItem[] {
    return [...population].sort((a, b) => {
        const rankA = rankings.get(a.id)!;
        const rankB = rankings.get(b.id)!;
        
        // First by rank (lower is better)
        if (rankA.rank !== rankB.rank) {
            return rankA.rank - rankB.rank;
        }
        
        // Then by crowding distance (higher is better)
        if (rankA.crowdingDistance !== rankB.crowdingDistance) {
            return rankB.crowdingDistance - rankA.crowdingDistance;
        }
        
        return 0;
    });
}
