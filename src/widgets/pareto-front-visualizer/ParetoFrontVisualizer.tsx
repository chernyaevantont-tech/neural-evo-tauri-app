import React, { useMemo, useState } from 'react';
import type { GenerationParetoFront, GenomeObjectives } from '../../shared/lib';
import { ParetoScatterPlot } from './ParetoScatterPlot';
import { ParetoSelector } from './ParetoSelector';
import styles from './pareto-front-visualizer.module.css';

export type ParetoViewMode = 'current' | 'global';

type Props = {
    currentParetoFront: GenomeObjectives[];
    paretoHistory: Map<number, GenerationParetoFront>;
    feasibilityByGenomeId?: Record<string, boolean>;
    constraintViolationScoreByGenomeId?: Record<string, number>;
    showOnlyFeasible?: boolean;
    onUseAsSeed?: (genomeId: string) => void;
    onOpenDetails?: (genomeId: string) => void;
    onExportSelected?: (genomeId: string) => void;
};

function dominates(a: GenomeObjectives, b: GenomeObjectives): boolean {
    const strictBetter =
        a.accuracy > b.accuracy ||
        a.inference_latency_ms < b.inference_latency_ms ||
        a.model_size_mb < b.model_size_mb;

    return (
        a.accuracy >= b.accuracy &&
        a.inference_latency_ms <= b.inference_latency_ms &&
        a.model_size_mb <= b.model_size_mb &&
        strictBetter
    );
}

function buildFrontier(genomes: GenomeObjectives[]): GenomeObjectives[] {
    return genomes.filter((candidate) =>
        !genomes.some(
            (other) => other.genome_id !== candidate.genome_id && dominates(other, candidate),
        ),
    );
}

function makeParetoPayload(
    generation: number,
    allGenomes: GenomeObjectives[],
    members: GenomeObjectives[],
): GenerationParetoFront {
    return {
        generation,
        total_genomes: allGenomes.length,
        pareto_members: members,
        objectives_3d: members.map((item) => [
            item.accuracy,
            item.inference_latency_ms,
            item.model_size_mb,
        ]),
        all_genomes: allGenomes,
        frontier_genome_ids: members.map((item) => item.genome_id),
    };
}

export function ParetoFrontVisualizer({
    currentParetoFront,
    paretoHistory,
    feasibilityByGenomeId,
    constraintViolationScoreByGenomeId,
    showOnlyFeasible = false,
    onUseAsSeed,
    onOpenDetails,
    onExportSelected,
}: Props) {
    const [viewMode, setViewMode] = useState<ParetoViewMode>('current');
    const [selectedGenomeId, setSelectedGenomeId] = useState<string>();

    const latestGeneration = useMemo(() => {
        if (paretoHistory.size === 0) {
            return 0;
        }
        return Math.max(...paretoHistory.keys());
    }, [paretoHistory]);

    const globalPareto = useMemo(() => {
        const uniq = new Map<string, GenomeObjectives>();
        for (const [generation, front] of paretoHistory.entries()) {
            const all = front.all_genomes ?? front.pareto_members;
            for (const genome of all) {
                const withGeneration = {
                    ...genome,
                    domination_count: genome.domination_count ?? 0,
                    is_dominated: genome.is_dominated ?? false,
                };
                const prev = uniq.get(genome.genome_id);
                if (!prev || generation >= latestGeneration) {
                    uniq.set(genome.genome_id, withGeneration);
                }
            }
        }

        const allGenomes = Array.from(uniq.values());
        const members = buildFrontier(allGenomes);
        return makeParetoPayload(latestGeneration, allGenomes, members);
    }, [latestGeneration, paretoHistory]);

    const currentPareto = useMemo(() => {
        const currentFront = paretoHistory.get(latestGeneration);
        if (currentFront) {
            return {
                ...currentFront,
                all_genomes: currentFront.all_genomes ?? currentFront.pareto_members,
                frontier_genome_ids:
                    currentFront.frontier_genome_ids ??
                    currentFront.pareto_members.map((item) => item.genome_id),
            };
        }
        return makeParetoPayload(latestGeneration, currentParetoFront, currentParetoFront);
    }, [currentParetoFront, latestGeneration, paretoHistory]);

    const rawDisplayPareto = viewMode === 'global' ? globalPareto : currentPareto;

    const displayPareto = useMemo(() => {
        const all = rawDisplayPareto.all_genomes ?? rawDisplayPareto.pareto_members;

        const withFeasibility = all.map((item) => {
            const feasible = feasibilityByGenomeId?.[item.genome_id] ?? item.device_feasible;
            const violation =
                constraintViolationScoreByGenomeId?.[item.genome_id] ?? item.constraint_violation_score;
            return {
                ...item,
                device_feasible: feasible,
                constraint_violation_score: violation,
            };
        });

        const feasibleOnly = showOnlyFeasible
            ? withFeasibility.filter((item) => item.device_feasible === true)
            : withFeasibility;

        const feasibleIds = new Set(feasibleOnly.map((item) => item.genome_id));
        const paretoMembers = (rawDisplayPareto.pareto_members ?? [])
            .map((item) => {
                const enriched = withFeasibility.find((candidate) => candidate.genome_id === item.genome_id);
                return enriched ?? item;
            })
            .filter((item) => !showOnlyFeasible || item.device_feasible === true);

        return {
            ...rawDisplayPareto,
            total_genomes: feasibleOnly.length,
            all_genomes: feasibleOnly,
            pareto_members: paretoMembers,
            frontier_genome_ids: (rawDisplayPareto.frontier_genome_ids ?? paretoMembers.map((item) => item.genome_id))
                .filter((id) => feasibleIds.has(id)),
        };
    }, [
        constraintViolationScoreByGenomeId,
        feasibilityByGenomeId,
        rawDisplayPareto,
        showOnlyFeasible,
    ]);

    const feasibilityCounts = useMemo(() => {
        const all = rawDisplayPareto.all_genomes ?? rawDisplayPareto.pareto_members;
        const feasible = all.filter((item) => {
            const state = feasibilityByGenomeId?.[item.genome_id] ?? item.device_feasible;
            return state === true;
        }).length;

        return {
            feasible,
            total: all.length,
        };
    }, [feasibilityByGenomeId, rawDisplayPareto]);

    const selectedGenome = useMemo(() => {
        if (!selectedGenomeId) {
            return undefined;
        }
        const all = displayPareto.all_genomes ?? displayPareto.pareto_members;
        return all.find((item) => item.genome_id === selectedGenomeId);
    }, [displayPareto.all_genomes, displayPareto.pareto_members, selectedGenomeId]);

    return (
        <div className={styles.card}>
            <div className={styles.headerRow}>
                <h3 className={styles.title}>Pareto Front</h3>
                <div className={styles.modeSwitch}>
                    <button
                        type="button"
                        className={`${styles.modeButton} ${viewMode === 'current' ? styles.modeButtonActive : ''}`}
                        onClick={() => setViewMode('current')}
                    >
                        Current generation
                    </button>
                    <button
                        type="button"
                        className={`${styles.modeButton} ${viewMode === 'global' ? styles.modeButtonActive : ''}`}
                        onClick={() => setViewMode('global')}
                    >
                        Global front
                    </button>
                </div>
            </div>

            <div className={styles.metaRow}>
                Generation: {displayPareto.generation} | Front size: {displayPareto.pareto_members.length} |
                Total: {(displayPareto.all_genomes ?? displayPareto.pareto_members).length} |
                Feasible: {feasibilityCounts.feasible}/{feasibilityCounts.total}
            </div>

            <ParetoScatterPlot
                pareto={displayPareto}
                selectedGenomeId={selectedGenomeId}
                onSelectGenome={setSelectedGenomeId}
                feasibilityByGenomeId={feasibilityByGenomeId}
                constraintViolationScoreByGenomeId={constraintViolationScoreByGenomeId}
            />

            <ParetoSelector
                selectedGenome={selectedGenome}
                onUseAsSeed={onUseAsSeed}
                onOpenDetails={onOpenDetails}
                onExportSelected={onExportSelected}
            />
        </div>
    );
}
