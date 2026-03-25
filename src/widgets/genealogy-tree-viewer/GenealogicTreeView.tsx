import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    getAncestors,
    getDescendants,
    getGenealogyPath,
    lineageRecordToGenomeGenealogy,
    type GenerationParetoFront,
    type GenomeGenealogy,
    type GenomeLineageRecordDto,
    type GenomeObjectives,
} from '../../shared/lib';
import { collectObjectives, useGenealogyGraph, type GenealogyGraphFilters, type GenealogyGraphNode } from '../../shared/hooks/useGenealogyGraph';
import { GenealogyFilters } from './GenealogyFilters';
import styles from './GenealogicTreeView.module.css';

type ViewTransform = {
    x: number;
    y: number;
    scale: number;
};

type Props = {
    genealogyTree?: Map<string, GenomeGenealogy>;
    paretoHistory?: Map<number, GenerationParetoFront>;
    onOpenGenomeDetails?: (genomeId: string) => void;
    onGenealogyTreeSync?: (tree: Map<string, GenomeGenealogy>) => void;
};

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 720;
const CANVAS_MARGIN_X = 90;
const CANVAS_MARGIN_Y = 80;

function mutationName(label: string): string {
    return label === 'Founder' ? 'Founder/Seed' : label;
}

function nodeToCanvas(node: GenealogyGraphNode) {
    return {
        x: CANVAS_MARGIN_X + node.x * (CANVAS_WIDTH - CANVAS_MARGIN_X * 2),
        y: CANVAS_MARGIN_Y + node.y * (CANVAS_HEIGHT - CANVAS_MARGIN_Y * 2),
    };
}

function clampScale(scale: number): number {
    return Math.min(2.5, Math.max(0.3, scale));
}

export function GenealogicTreeView({
    genealogyTree,
    paretoHistory = new Map(),
    onOpenGenomeDetails,
    onGenealogyTreeSync,
}: Props) {
    const wrapperRef = useRef<HTMLDivElement | null>(null);
    const [selectedGenomeId, setSelectedGenomeId] = useState<string | undefined>(undefined);
    const [comparedParentId, setComparedParentId] = useState<string | undefined>(undefined);
    const [transform, setTransform] = useState<ViewTransform>({ x: 0, y: 0, scale: 1 });

    const [isPanning, setIsPanning] = useState(false);
    const [apiLoading, setApiLoading] = useState(false);
    const [apiError, setApiError] = useState<string | undefined>(undefined);
    const [backendAncestors, setBackendAncestors] = useState<GenomeLineageRecordDto[]>([]);
    const [backendDescendants, setBackendDescendants] = useState<GenomeLineageRecordDto[]>([]);
    const panStartRef = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null);

    const bounds = useMemo(() => {
        if (!genealogyTree || genealogyTree.size === 0) {
            return {
                generation: { min: 0, max: 0 },
                fitness: { min: 0, max: 1 },
            };
        }

        const generations = Array.from(genealogyTree.values()).map((g) => g.generation);
        const fitnessValues = Array.from(genealogyTree.values()).map((g) => g.fitness);

        return {
            generation: { min: Math.min(...generations), max: Math.max(...generations) },
            fitness: { min: Math.min(...fitnessValues), max: Math.max(...fitnessValues) },
        };
    }, [genealogyTree]);

    const [filters, setFilters] = useState<GenealogyGraphFilters>({
        generationMin: bounds.generation.min,
        generationMax: bounds.generation.max,
        fitnessMin: bounds.fitness.min,
        fitnessMax: bounds.fitness.max,
        paretoOnly: false,
        ancestorsDepth: 1,
    });

    useEffect(() => {
        setFilters((prev) => ({
            ...prev,
            generationMin: bounds.generation.min,
            generationMax: bounds.generation.max,
            fitnessMin: bounds.fitness.min,
            fitnessMax: bounds.fitness.max,
        }));
    }, [bounds.fitness.max, bounds.fitness.min, bounds.generation.max, bounds.generation.min]);

    const paretoGenomeIds = useMemo(() => {
        const ids = new Set<string>();
        for (const front of paretoHistory.values()) {
            const source = front.frontier_genome_ids ?? front.pareto_members.map((item) => item.genome_id);
            source.forEach((id) => ids.add(id));
        }
        return ids;
    }, [paretoHistory]);

    const objectivesByGenomeId = useMemo<Map<string, GenomeObjectives>>(
        () => collectObjectives(paretoHistory),
        [paretoHistory],
    );

    const { nodes, edges, selectedNode, generationBounds, fitnessBounds } = useGenealogyGraph({
        genealogyTree,
        filters,
        selectedGenomeId,
        paretoGenomeIds,
        objectivesByGenomeId,
    });

    const nodesById = useMemo(() => {
        const map = new Map<string, GenealogyGraphNode>();
        nodes.forEach((node) => map.set(node.id, node));
        return map;
    }, [nodes]);

    const fitToScreen = useCallback(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) {
            return;
        }

        if (nodes.length === 0) {
            setTransform({ x: 0, y: 0, scale: 1 });
            return;
        }

        const rect = wrapper.getBoundingClientRect();
        const points = nodes.map((node) => nodeToCanvas(node));

        const minX = Math.min(...points.map((p) => p.x));
        const maxX = Math.max(...points.map((p) => p.x));
        const minY = Math.min(...points.map((p) => p.y));
        const maxY = Math.max(...points.map((p) => p.y));

        const contentWidth = Math.max(1, maxX - minX);
        const contentHeight = Math.max(1, maxY - minY);

        const padding = 50;
        const scaleX = (rect.width - padding) / contentWidth;
        const scaleY = (rect.height - padding) / contentHeight;
        const scale = clampScale(Math.min(scaleX, scaleY));

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        setTransform({
            scale,
            x: rect.width / 2 - centerX * scale,
            y: rect.height / 2 - centerY * scale,
        });
    }, [nodes]);

    const centerOnSelected = useCallback(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper || !selectedNode) {
            return;
        }

        const rect = wrapper.getBoundingClientRect();
        const point = nodeToCanvas(selectedNode);
        setTransform((prev) => ({
            ...prev,
            x: rect.width / 2 - point.x * prev.scale,
            y: rect.height / 2 - point.y * prev.scale,
        }));
    }, [selectedNode]);

    useEffect(() => {
        fitToScreen();
    }, [fitToScreen]);

    const selectedParentCandidates = selectedNode
        ? selectedNode.parentIds.map((id) => nodesById.get(id)).filter((n): n is GenealogyGraphNode => Boolean(n))
        : [];

    const comparedParent = comparedParentId ? nodesById.get(comparedParentId) : undefined;

    const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
        setIsPanning(true);
        panStartRef.current = {
            x: event.clientX,
            y: event.clientY,
            originX: transform.x,
            originY: transform.y,
        };
    };

    const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
        if (!isPanning || !panStartRef.current) {
            return;
        }

        const dx = event.clientX - panStartRef.current.x;
        const dy = event.clientY - panStartRef.current.y;

        setTransform((prev) => ({
            ...prev,
            x: (panStartRef.current?.originX ?? prev.x) + dx,
            y: (panStartRef.current?.originY ?? prev.y) + dy,
        }));
    };

    const stopPanning = () => {
        setIsPanning(false);
        panStartRef.current = null;
    };

    const onWheel = (event: React.WheelEvent<SVGSVGElement>) => {
        event.preventDefault();
        const wrapper = wrapperRef.current;
        if (!wrapper) {
            return;
        }

        const rect = wrapper.getBoundingClientRect();
        const pointerX = event.clientX - rect.left;
        const pointerY = event.clientY - rect.top;
        const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;

        setTransform((prev) => {
            const nextScale = clampScale(prev.scale * zoomFactor);
            const worldX = (pointerX - prev.x) / prev.scale;
            const worldY = (pointerY - prev.y) / prev.scale;

            return {
                scale: nextScale,
                x: pointerX - worldX * nextScale,
                y: pointerY - worldY * nextScale,
            };
        });
    };

    const selectedObjectives = selectedNode?.objectives;
    const comparedObjectives = comparedParent?.objectives;

    const mergeRecordsToTree = useCallback((records: GenomeLineageRecordDto[]) => {
        const next = new Map(genealogyTree ?? new Map<string, GenomeGenealogy>());
        for (const record of records) {
            const prev = next.get(record.genome_id);
            next.set(record.genome_id, {
                ...lineageRecordToGenomeGenealogy(record),
                fitness: prev?.fitness ?? 0,
                accuracy: prev?.accuracy ?? 0,
            });
        }
        onGenealogyTreeSync?.(next);
    }, [genealogyTree, onGenealogyTreeSync]);

    const syncPathFromBackend = useCallback(async () => {
        if (!selectedNode) {
            return;
        }

        setApiLoading(true);
        setApiError(undefined);
        try {
            const path = await getGenealogyPath(selectedNode.id);
            mergeRecordsToTree(path.records);
        } catch (error) {
            setApiError(String(error));
        } finally {
            setApiLoading(false);
        }
    }, [mergeRecordsToTree, selectedNode]);

    const loadAncestorsFromBackend = useCallback(async () => {
        if (!selectedNode) {
            return;
        }

        setApiLoading(true);
        setApiError(undefined);
        try {
            const records = await getAncestors(selectedNode.id, filters.ancestorsDepth);
            setBackendAncestors(records);
            mergeRecordsToTree(records);
        } catch (error) {
            setApiError(String(error));
        } finally {
            setApiLoading(false);
        }
    }, [filters.ancestorsDepth, mergeRecordsToTree, selectedNode]);

    const loadDescendantsFromBackend = useCallback(async () => {
        if (!selectedNode) {
            return;
        }

        setApiLoading(true);
        setApiError(undefined);
        try {
            const records = await getDescendants(selectedNode.id, filters.ancestorsDepth);
            setBackendDescendants(records);
            mergeRecordsToTree(records);
        } catch (error) {
            setApiError(String(error));
        } finally {
            setApiLoading(false);
        }
    }, [filters.ancestorsDepth, mergeRecordsToTree, selectedNode]);

    return (
        <section className={styles.card}>
            <div className={styles.headerRow}>
                <h3 className={styles.title}>Genealogy</h3>
                <div className={styles.actionRow}>
                    <button type="button" className={styles.actionButton} onClick={fitToScreen}>Fit</button>
                    <button type="button" className={styles.actionButton} onClick={centerOnSelected} disabled={!selectedNode}>
                        Center selected
                    </button>
                </div>
            </div>

            <GenealogyFilters
                filters={filters}
                generationBounds={generationBounds}
                fitnessBounds={fitnessBounds}
                onChange={setFilters}
            />

            <div className={styles.meta}>
                Visible nodes: {nodes.length} | Visible edges: {edges.length}
            </div>

            <div className={styles.canvasWrap} ref={wrapperRef}>
                {nodes.length === 0 ? (
                    <div className={styles.emptyState}>No genealogy nodes match current filters.</div>
                ) : (
                    <svg
                        width="100%"
                        height="100%"
                        viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={stopPanning}
                        onPointerLeave={stopPanning}
                        onWheel={onWheel}
                    >
                        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
                            {edges.map((edge) => {
                                const source = nodesById.get(edge.source);
                                const target = nodesById.get(edge.target);
                                if (!source || !target) {
                                    return null;
                                }

                                const s = nodeToCanvas(source);
                                const t = nodeToCanvas(target);
                                const labelX = (s.x + t.x) / 2;
                                const labelY = (s.y + t.y) / 2 - 8;

                                return (
                                    <g key={edge.id}>
                                        <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} className={styles.edgeLine} />
                                        <text x={labelX} y={labelY} className={styles.edgeLabel}>{edge.label}</text>
                                    </g>
                                );
                            })}

                            {nodes.map((node) => {
                                const point = nodeToCanvas(node);
                                const isSelected = node.id === selectedGenomeId;

                                return (
                                    <g
                                        key={node.id}
                                        className={styles.nodeGroup}
                                        data-testid={`genealogy-node-${node.id}`}
                                        onClick={() => {
                                            setSelectedGenomeId(node.id);
                                            setComparedParentId(undefined);
                                        }}
                                    >
                                        <circle
                                            cx={point.x}
                                            cy={point.y}
                                            r={isSelected ? 24 : 18}
                                            className={`${styles.nodeCircle} ${node.isPareto ? styles.nodePareto : ''} ${isSelected ? styles.nodeSelected : ''}`}
                                        />
                                        <text x={point.x} y={point.y + 4} textAnchor="middle" className={styles.nodeText}>
                                            {node.id}
                                        </text>
                                        <text x={point.x} y={point.y - 28} textAnchor="middle" className={styles.nodeSubText}>
                                            G{node.generation}
                                        </text>
                                    </g>
                                );
                            })}
                        </g>
                    </svg>
                )}
            </div>

            <div className={styles.detailsCard}>
                {!selectedNode ? (
                    <div className={styles.emptyDetails}>Select a genome node to inspect lineage details.</div>
                ) : (
                    <>
                        <div className={styles.detailsTitle}>Genome {selectedNode.id}</div>
                        <div className={styles.detailsGrid}>
                            <span>Generation: {selectedNode.generation}</span>
                            <span>Mutation: {mutationName(selectedNode.mutationLabel)}</span>
                            <span>Fitness: {selectedNode.fitness.toFixed(4)}</span>
                            <span>Accuracy: {selectedNode.accuracy.toFixed(4)}</span>
                            <span>
                                Latency: {selectedObjectives?.inference_latency_ms?.toFixed(3) ?? 'N/A'} ms
                            </span>
                            <span>Model size: {selectedObjectives?.model_size_mb?.toFixed(3) ?? 'N/A'} MB</span>
                        </div>

                        <div className={styles.parentSection}>
                            <div className={styles.parentTitle}>Parents</div>
                            {selectedParentCandidates.length === 0 ? (
                                <div className={styles.parentEmpty}>Founder node has no parents.</div>
                            ) : (
                                <div className={styles.parentList}>
                                    {selectedParentCandidates.map((parent) => (
                                        <button
                                            key={parent.id}
                                            type="button"
                                            className={styles.parentButton}
                                            onClick={() => setComparedParentId(parent.id)}
                                        >
                                            Compare with parent {parent.id}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {comparedParent && (
                            <div className={styles.compareBox}>
                                <div className={styles.compareTitle}>Comparison: {selectedNode.id} vs {comparedParent.id}</div>
                                <div className={styles.detailsGrid}>
                                    <span>Fitness delta: {(selectedNode.fitness - comparedParent.fitness).toFixed(4)}</span>
                                    <span>Accuracy delta: {(selectedNode.accuracy - comparedParent.accuracy).toFixed(4)}</span>
                                    <span>
                                        Latency delta: {selectedObjectives && comparedObjectives
                                            ? (selectedObjectives.inference_latency_ms - comparedObjectives.inference_latency_ms).toFixed(3)
                                            : 'N/A'}
                                    </span>
                                    <span>
                                        Size delta: {selectedObjectives && comparedObjectives
                                            ? (selectedObjectives.model_size_mb - comparedObjectives.model_size_mb).toFixed(3)
                                            : 'N/A'}
                                    </span>
                                </div>
                            </div>
                        )}

                        <div className={styles.actionRow}>
                            <button
                                type="button"
                                className={styles.actionButton}
                                onClick={syncPathFromBackend}
                                disabled={apiLoading}
                            >
                                Sync path (API)
                            </button>
                            <button
                                type="button"
                                className={styles.actionButton}
                                onClick={loadAncestorsFromBackend}
                                disabled={apiLoading}
                            >
                                Load ancestors (API)
                            </button>
                            <button
                                type="button"
                                className={styles.actionButton}
                                onClick={loadDescendantsFromBackend}
                                disabled={apiLoading}
                            >
                                Load descendants (API)
                            </button>
                            <button
                                type="button"
                                className={styles.actionButton}
                                onClick={() => onOpenGenomeDetails?.(selectedNode.id)}
                            >
                                Open details
                            </button>
                        </div>

                        <div className={styles.backendMeta}>
                            API ancestors: {backendAncestors.length} | API descendants: {backendDescendants.length}
                            {apiError ? ` | API error: ${apiError}` : ''}
                        </div>
                    </>
                )}
            </div>
        </section>
    );
}
