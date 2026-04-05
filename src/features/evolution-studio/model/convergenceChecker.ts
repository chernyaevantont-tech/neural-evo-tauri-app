/**
 * Convergence Checker for Multi-Objective Evolution
 * 
 * Determines when to stop evolution based on:
 * - Hypervolume improvement (convergence)
 * - Target quality reached
 * - Stability of Pareto front
 * 
 * @see multiObjectiveFitness.ts - ObjectiveVector type
 */

import { ObjectiveVector } from './multiObjectiveFitness';

/**
 * Convergence check result
 */
export interface ConvergenceReport {
    shouldStop: boolean;
    reason: 'convergence' | 'target_reached' | null;
    metrics: {
        hypervolume: number;
        hypervolumeImprovement: number;
        generationsWithoutImprovement: number;
        frontSize: number;
        bestQuality: number;
    };
}

/**
 * Convergence criteria configuration
 */
export interface ConvergenceCriteria {
    useHypervolumeConvergence: boolean;
    hypervolumeThreshold: number;     // Minimum improvement to continue (e.g., 0.001)
    convergencePatience: number;       // Generations without improvement before stopping
    useTargetQuality: boolean;
    targetQuality: number;             // Stop if any solution reaches this quality
}

/**
 * Convergence checker class
 * 
 * Tracks hypervolume history and front stability across generations
 */
export class ConvergenceChecker {
    private hypervolumeHistory: number[] = [];
    private bestQualityHistory: number[] = [];
    
    constructor(private criteria: ConvergenceCriteria) {}
    
    /**
     * Check if evolution should stop based on convergence
     * 
     * @param paretoFront - Current Pareto front solutions
     * @param generation - Current generation number
     * @returns Convergence report with decision and metrics
     */
    checkConvergence(
        paretoFront: Array<{ vector: ObjectiveVector }>
    ): ConvergenceReport {
        // Calculate hypervolume
        const hypervolume = this.calculateHypervolume(paretoFront);
        const prevHypervolume = this.hypervolumeHistory[this.hypervolumeHistory.length - 1] || 0;
        const hypervolumeImprovement = hypervolume - prevHypervolume;
        
        // Track history
        this.hypervolumeHistory.push(hypervolume);
        
        // Track best quality
        const bestQuality = Math.max(...paretoFront.map(p => p.vector.quality));
        this.bestQualityHistory.push(bestQuality);
        
        // Check stopping criteria
        let shouldStop = false;
        let reason: ConvergenceReport['reason'] = null;
        
        // 1. Hypervolume convergence
        if (this.criteria.useHypervolumeConvergence) {
            if (hypervolumeImprovement < this.criteria.hypervolumeThreshold) {
                const generationsWithoutImprovement = this.countGenerationsWithoutImprovement();
                
                if (generationsWithoutImprovement >= this.criteria.convergencePatience) {
                    shouldStop = true;
                    reason = 'convergence';
                }
            }
        }
        
        // 2. Target quality reached
        if (this.criteria.useTargetQuality && bestQuality >= this.criteria.targetQuality) {
            shouldStop = true;
            reason = 'target_reached';
        }
        
        return {
            shouldStop,
            reason,
            metrics: {
                hypervolume,
                hypervolumeImprovement,
                generationsWithoutImprovement: this.countGenerationsWithoutImprovement(),
                frontSize: paretoFront.length,
                bestQuality
            }
        };
    }
    
    /**
     * Calculate hypervolume of Pareto front
     * 
     * Simplified 2D hypervolume using:
     * - Quality (maximize)
     * - Normalized resources (minimize)
     * 
     * Reference point: (0, 1) - worst possible values
     */
    private calculateHypervolume(front: Array<{ vector: ObjectiveVector }>): number {
        if (front.length === 0) return 0;
        
        const referencePoint = { quality: 0, resources: 1 };
        
        // Normalize resources to [0, 1]
        const normalized = front.map(p => ({
            quality: p.vector.quality,
            resources: this.normalizeResources(p.vector)
        }));
        
        // Sort by quality descending
        const sorted = [...normalized].sort((a, b) => b.quality - a.quality);
        
        let hypervolume = 0;
        let prevResources = referencePoint.resources;
        
        for (const solution of sorted) {
            // Only count if better than reference
            if (solution.quality > referencePoint.quality && 
                solution.resources < referencePoint.resources) {
                
                hypervolume += (solution.quality - referencePoint.quality) * 
                              (prevResources - solution.resources);
                prevResources = solution.resources;
            }
        }
        
        return Math.max(0, hypervolume);
    }
    
    /**
     * Normalize resources to [0, 1] range
     */
    private normalizeResources(vector: ObjectiveVector): number {
        // Normalize each resource to [0, 1] based on typical ranges
        const flashNorm = Math.min(1, vector.flashKB / 10000);    // 10MB max
        const ramNorm = Math.min(1, vector.ramKB / 5000);         // 5MB max
        const macsNorm = Math.min(1, vector.macs / 100);          // 100M MACs max
        
        // Average of normalized resources (lower is better)
        return (flashNorm + ramNorm + macsNorm) / 3;
    }
    
    /**
     * Count generations without significant hypervolume improvement
     */
    private countGenerationsWithoutImprovement(): number {
        let count = 0;
        
        for (let i = this.hypervolumeHistory.length - 2; i >= 0; i--) {
            const improvement = this.hypervolumeHistory[i + 1] - this.hypervolumeHistory[i];
            
            if (improvement < this.criteria.hypervolumeThreshold) {
                count++;
            } else {
                break;
            }
        }
        
        return count;
    }
    
    /**
     * Reset checker state (for new evolution run)
     */
    reset(): void {
        this.hypervolumeHistory = [];
        this.bestQualityHistory = [];
    }
}
