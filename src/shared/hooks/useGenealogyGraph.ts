import { useMemo } from 'react';
import { forceCollide, forceManyBody, forceSimulation, forceX, forceY } from 'd3-force';
import type { GenerationParetoFront, GenomeGenealogy, GenomeObjectives, MutationType } from '../lib';

export interface GenealogyGraphFilters {
    generationMin: number;
    generationMax: number;
    fitnessMin: number;
    fitnessMax: number;
    paretoOnly: boolean;
    ancestorsDepth: number;
}

export interface GenealogyGraphNode {
    id: string;
    generation: number;
    fitness: number;
    accuracy: number;
    parentIds: string[];
    mutationLabel: string;
    x: number;
    y: number;
    isPareto: boolean;
    createdAtMs: number;
    objectives?: GenomeObjectives;
}

export interface GenealogyGraphEdge {
    id: string;
    source: string;
    target: string;
    label: string;
}

export interface GenealogyGraphResult {
    nodes: GenealogyGraphNode[];
    edges: GenealogyGraphEdge[];
    selectedNode?: GenealogyGraphNode;
    generationBounds: { min: number; max: number };
    fitnessBounds: { min: number; max: number };
}

interface SimNode {
    id: string;
    generation: number;
    x: number;
    y: number;
    vx?: number;
    vy?: number;
}

interface UseGenealogyGraphParams {
    genealogyTree?: Map<string, GenomeGenealogy>;
    filters: GenealogyGraphFilters;
    selectedGenomeId?: string;
    paretoGenomeIds?: Set<string>;
    objectivesByGenomeId?: Map<string, GenomeObjectives>;
}

function mutationTypeLabel(mutation: MutationType): string {
    switch (mutation.type) {
        case 'AddNode':
        case 'RemoveNode':
        case 'RemoveSubgraph':
        case 'ParameterMutation':
        case 'ParameterScale':
        case 'Crossover':
            return mutation.type;
        case 'Random':
        default:
            return 'Founder';
    }
}

function getBounds(genealogyTree?: Map<string, GenomeGenealogy>) {
    if (!genealogyTree || genealogyTree.size === 0) {
        return {
            generation: { min: 0, max: 0 },
            fitness: { min: 0, max: 0 },
        };
    }

    const generations = Array.from(genealogyTree.values()).map((g) => g.generation);
    const fitnessValues = Array.from(genealogyTree.values()).map((g) => g.fitness);

    return {
        generation: {
            min: Math.min(...generations),
            max: Math.max(...generations),
        },
        fitness: {
            min: Math.min(...fitnessValues),
            max: Math.max(...fitnessValues),
        },
    };
}

export function collectObjectives(paretoHistory: Map<number, GenerationParetoFront>): Map<string, GenomeObjectives> {
    const map = new Map<string, GenomeObjectives>();

    for (const front of paretoHistory.values()) {
        for (const item of front.all_genomes ?? front.pareto_members) {
            map.set(item.genome_id, item);
        }
    }

    return map;
}

function applyD3Layout(rawNodes: Omit<GenealogyGraphNode, 'x' | 'y'>[]): Map<string, { x: number; y: number }> {
    if (rawNodes.length === 0) {
        return new Map();
    }

    const generations = Array.from(new Set(rawNodes.map((node) => node.generation))).sort((a, b) => a - b);
    const minGen = generations[0];
    const maxGen = generations[generations.length - 1];
    const genSpan = Math.max(1, maxGen - minGen);

    const groupedByGeneration = new Map<number, Omit<GenealogyGraphNode, 'x' | 'y'>[]>();
    for (const node of rawNodes) {
        const group = groupedByGeneration.get(node.generation) ?? [];
        group.push(node);
        groupedByGeneration.set(node.generation, group);
    }

    for (const group of groupedByGeneration.values()) {
        group.sort((a, b) => {
            if (a.createdAtMs !== b.createdAtMs) {
                return a.createdAtMs - b.createdAtMs;
            }
            return a.id.localeCompare(b.id);
        });
    }

    const simNodes: SimNode[] = rawNodes.map((node, idx) => ({
        id: node.id,
        generation: node.generation,
        x: (idx % 10) * 20,
        y: ((node.generation - minGen) / genSpan) * 400,
    }));

    const targetXById = new Map<string, number>();
    for (const [generation, group] of groupedByGeneration.entries()) {
        group.forEach((node, idx) => {
            const localX = (idx + 1) / (group.length + 1);
            targetXById.set(node.id, localX * 600);
        });

        if (group.length === 0) {
            targetXById.set(String(generation), 300);
        }
    }

    const simulation = forceSimulation(simNodes)
        .force('x', forceX<SimNode>((node: SimNode) => targetXById.get(node.id) ?? 300).strength(0.25))
        .force('y', forceY<SimNode>((node: SimNode) => ((node.generation - minGen) / genSpan) * 400).strength(0.95))
        .force('charge', forceManyBody().strength(-45))
        .force('collide', forceCollide(18))
        .stop();

    for (let i = 0; i < 140; i += 1) {
        simulation.tick();
    }

    const simulatedByGeneration = new Map<number, SimNode[]>();
    for (const node of simNodes) {
        const list = simulatedByGeneration.get(node.generation) ?? [];
        list.push(node);
        simulatedByGeneration.set(node.generation, list);
    }

    const result = new Map<string, { x: number; y: number }>();
    for (const generation of generations) {
        const list = simulatedByGeneration.get(generation) ?? [];
        list.sort((a, b) => (a.x ?? 0) - (b.x ?? 0));
        const y = genSpan === 0 ? 0.5 : (generation - minGen) / genSpan;

        list.forEach((node, idx) => {
            const x = (idx + 1) / (list.length + 1);
            result.set(node.id, { x, y });
        });
    }

    return result;
}

export function useGenealogyGraph({
    genealogyTree,
    filters,
    selectedGenomeId,
    paretoGenomeIds,
    objectivesByGenomeId,
}: UseGenealogyGraphParams): GenealogyGraphResult {
    return useMemo(() => {
        const bounds = getBounds(genealogyTree);

        if (!genealogyTree || genealogyTree.size === 0) {
            return {
                nodes: [],
                edges: [],
                selectedNode: undefined,
                generationBounds: bounds.generation,
                fitnessBounds: bounds.fitness,
            };
        }

        const visibleIds = new Set<string>();

        for (const [id, node] of genealogyTree.entries()) {
            const inGenerationRange =
                node.generation >= filters.generationMin && node.generation <= filters.generationMax;
            const inFitnessRange = node.fitness >= filters.fitnessMin && node.fitness <= filters.fitnessMax;
            const passesPareto = !filters.paretoOnly || Boolean(paretoGenomeIds?.has(id));

            if (inGenerationRange && inFitnessRange && passesPareto) {
                visibleIds.add(id);
            }
        }

        if (selectedGenomeId && genealogyTree.has(selectedGenomeId)) {
            visibleIds.add(selectedGenomeId);

            if (filters.ancestorsDepth > 0) {
                const queue: Array<{ id: string; depth: number }> = [{ id: selectedGenomeId, depth: 0 }];
                const visited = new Set<string>([selectedGenomeId]);

                while (queue.length > 0) {
                    const current = queue.shift();
                    if (!current) {
                        continue;
                    }

                    const node = genealogyTree.get(current.id);
                    if (!node || current.depth >= filters.ancestorsDepth) {
                        continue;
                    }

                    for (const parentId of node.parent_ids) {
                        if (!genealogyTree.has(parentId) || visited.has(parentId)) {
                            continue;
                        }

                        visibleIds.add(parentId);
                        visited.add(parentId);
                        queue.push({ id: parentId, depth: current.depth + 1 });
                    }
                }
            }
        }

        const rawNodes: Omit<GenealogyGraphNode, 'x' | 'y'>[] = [];
        for (const id of visibleIds) {
            const node = genealogyTree.get(id);
            if (!node) {
                continue;
            }

            rawNodes.push({
                id: node.genome_id,
                generation: node.generation,
                fitness: node.fitness,
                accuracy: node.accuracy,
                parentIds: node.parent_ids,
                mutationLabel: mutationTypeLabel(node.mutation_type),
                isPareto: Boolean(paretoGenomeIds?.has(node.genome_id)),
                createdAtMs: node.created_at_ms,
                objectives: objectivesByGenomeId?.get(node.genome_id),
            });
        }

        const coordinates = applyD3Layout(rawNodes);

        const nodes: GenealogyGraphNode[] = [];
        for (const node of rawNodes) {
            const point = coordinates.get(node.id) ?? { x: 0.5, y: 0.5 };
            nodes.push({ ...node, x: point.x, y: point.y });
        }

        const visibleIdSet = new Set(nodes.map((node) => node.id));
        const edges: GenealogyGraphEdge[] = [];

        for (const child of nodes) {
            for (const parentId of child.parentIds) {
                if (!visibleIdSet.has(parentId)) {
                    continue;
                }

                edges.push({
                    id: `${parentId}->${child.id}`,
                    source: parentId,
                    target: child.id,
                    label: child.mutationLabel,
                });
            }
        }

        return {
            nodes,
            edges,
            selectedNode: nodes.find((node) => node.id === selectedGenomeId),
            generationBounds: bounds.generation,
            fitnessBounds: bounds.fitness,
        };
    }, [filters, genealogyTree, objectivesByGenomeId, paretoGenomeIds, selectedGenomeId]);
}
