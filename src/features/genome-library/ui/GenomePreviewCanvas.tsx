import React, { useEffect, useState, useRef } from 'react';
import { useGenomeLibraryStore } from '../model/store';
import { deserializeGenome } from '../../../entities/canvas-genome/lib/deserializeGenome';
import { BaseNode } from '../../../entities/canvas-genome/model/nodes/base_node';
import { getNodeColor, getNodeLabel, theme } from '../../../shared/lib';

interface Position {
    x: number;
    y: number;
}

interface PreviewNode {
    id: string;
    type: string;
    position: Position;
}

interface PreviewConnection {
    fromId: string;
    toId: string;
}

/**
 * Simplified force-directed layout for preview (centered at origin).
 */
function computeLayout(nodes: BaseNode[]): Map<string, Position> {
    if (nodes.length === 0) return new Map();

    const positions = new Map<string, Position>();
    const velocities = new Map<string, Position>();

    // Initial positions in a circle
    nodes.forEach((node, index) => {
        const angle = (index / nodes.length) * Math.PI * 2;
        const radius = 80;
        positions.set(node.id, {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
        });
        velocities.set(node.id, { x: 0, y: 0 });
    });

    const iterations = 200;
    const REPULSION = 4000;
    const ATTRACTION = 0.015;
    const DAMPING = 0.85;
    const IDEAL_DIST = 120;

    for (let iter = 0; iter < iterations; iter++) {
        const forces = new Map<string, Position>();
        nodes.forEach(n => forces.set(n.id, { x: 0, y: 0 }));

        // Repulsion
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const p1 = positions.get(nodes[i].id)!;
                const p2 = positions.get(nodes[j].id)!;
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = REPULSION / (dist * dist);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                forces.get(nodes[i].id)!.x -= fx;
                forces.get(nodes[i].id)!.y -= fy;
                forces.get(nodes[j].id)!.x += fx;
                forces.get(nodes[j].id)!.y += fy;
            }
        }

        // Attraction along edges
        nodes.forEach(node => {
            node.next.forEach(nextNode => {
                const p1 = positions.get(node.id);
                const p2 = positions.get(nextNode.id);
                if (!p1 || !p2) return;
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = ATTRACTION * (dist - IDEAL_DIST);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                forces.get(node.id)!.x += fx;
                forces.get(node.id)!.y += fy;
                forces.get(nextNode.id)!.x -= fx;
                forces.get(nextNode.id)!.y -= fy;
            });
        });

        // Apply
        nodes.forEach(node => {
            const v = velocities.get(node.id)!;
            const f = forces.get(node.id)!;
            const p = positions.get(node.id)!;
            v.x = (v.x + f.x) * DAMPING;
            v.y = (v.y + f.y) * DAMPING;
            p.x += v.x;
            p.y += v.y;
        });
    }

    return positions;
}

interface GenomePreviewCanvasProps {
    genomeId: string;
}

export const GenomePreviewCanvas: React.FC<GenomePreviewCanvasProps> = ({ genomeId }) => {
    const loadGenomeContent = useGenomeLibraryStore(s => s.loadGenomeContent);
    const [previewNodes, setPreviewNodes] = useState<PreviewNode[]>([]);
    const [connections, setConnections] = useState<PreviewConnection[]>([]);
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    // Pan & Zoom state
    const [translate, setTranslate] = useState({ x: 0, y: 0 });
    const [scale, setScale] = useState(1);
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });

    useEffect(() => {
        let cancelled = false;

        const load = async () => {
            setLoading(true);
            try {
                const genomeStr = await loadGenomeContent(genomeId);
                const { nodes } = await deserializeGenome(genomeStr);

                if (cancelled) return;

                const positions = computeLayout(nodes);

                const pNodes: PreviewNode[] = nodes.map(n => ({
                    id: n.id,
                    type: n.GetNodeType(),
                    position: positions.get(n.id) || { x: 0, y: 0 },
                }));

                const conns: PreviewConnection[] = [];
                nodes.forEach(n => {
                    n.next.forEach(nextNode => {
                        conns.push({ fromId: n.id, toId: nextNode.id });
                    });
                });

                setPreviewNodes(pNodes);
                setConnections(conns);

                // Auto-fit: center the graph in the container
                if (pNodes.length > 0 && containerRef.current) {
                    const nodeRadius = 35;
                    const padding = 60;
                    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                    pNodes.forEach(n => {
                        minX = Math.min(minX, n.position.x - nodeRadius);
                        maxX = Math.max(maxX, n.position.x + nodeRadius);
                        minY = Math.min(minY, n.position.y - nodeRadius);
                        maxY = Math.max(maxY, n.position.y + nodeRadius);
                    });

                    const graphW = maxX - minX + padding * 2;
                    const graphH = maxY - minY + padding * 2;
                    const containerW = containerRef.current.clientWidth;
                    const containerH = containerRef.current.clientHeight;

                    const fitScale = Math.min(containerW / graphW, containerH / graphH, 1.5);
                    const graphCenterX = (minX + maxX) / 2;
                    const graphCenterY = (minY + maxY) / 2;

                    setScale(fitScale);
                    setTranslate({
                        x: containerW / 2 - graphCenterX * fitScale,
                        y: containerH / 2 - graphCenterY * fitScale,
                    });
                }
            } catch (err) {
                console.error('Preview load failed:', err);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        load();
        return () => { cancelled = true; };
    }, [genomeId, loadGenomeContent]);

    const onMouseDown = (e: React.MouseEvent) => {
        if (e.button === 2) {
            e.preventDefault();
            setIsPanning(true);
            setPanStart({ x: e.clientX - translate.x, y: e.clientY - translate.y });
        }
    };

    const onMouseMove = (e: React.MouseEvent) => {
        if (isPanning) {
            setTranslate({
                x: e.clientX - panStart.x,
                y: e.clientY - panStart.y,
            });
        }
    };

    const onMouseUp = () => {
        setIsPanning(false);
    };

    const onWheel = (e: React.WheelEvent) => {
        e.stopPropagation();
        const rect = svgRef.current?.getBoundingClientRect();
        if (!rect) return;

        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const newScale = Math.max(0.1, Math.min(5, scale * zoomFactor));

        // Zoom towards mouse position
        setTranslate({
            x: mouseX - (mouseX - translate.x) * (newScale / scale),
            y: mouseY - (mouseY - translate.y) * (newScale / scale),
        });
        setScale(newScale);
    };

    if (loading) {
        return (
            <div style={{
                width: '100%', height: '100%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                opacity: 0.5,
            }}>
                Loading preview...
            </div>
        );
    }

    const nodeRadius = 35;

    return (
        <div
            ref={containerRef}
            style={{ width: '100%', height: '100%', overflow: 'hidden' }}
        >
            <svg
                ref={svgRef}
                width="100%"
                height="100%"
                style={{
                    background: theme.colors.background.canvas,
                    cursor: isPanning ? 'grabbing' : 'default',
                }}
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseUp}
                onWheel={onWheel}
                onContextMenu={e => e.preventDefault()}
            >
                <g transform={`translate(${translate.x}, ${translate.y}) scale(${scale})`}>
                    {/* Connections */}
                    {connections.map((conn, i) => {
                        const from = previewNodes.find(n => n.id === conn.fromId);
                        const to = previewNodes.find(n => n.id === conn.toId);
                        if (!from || !to) return null;

                        const dx = to.position.x - from.position.x;
                        const dy = to.position.y - from.position.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const angle = Math.atan2(dy, dx);

                        if (dist < nodeRadius * 2) return null;

                        const startX = from.position.x + nodeRadius * Math.cos(angle);
                        const startY = from.position.y + nodeRadius * Math.sin(angle);
                        const endX = to.position.x - nodeRadius * Math.cos(angle);
                        const endY = to.position.y - nodeRadius * Math.sin(angle);

                        const arrowSize = 8;
                        const arrowTip = { x: endX, y: endY };
                        const arrowBase1 = {
                            x: endX - arrowSize * Math.cos(angle - Math.PI / 6),
                            y: endY - arrowSize * Math.sin(angle - Math.PI / 6),
                        };
                        const arrowBase2 = {
                            x: endX - arrowSize * Math.cos(angle + Math.PI / 6),
                            y: endY - arrowSize * Math.sin(angle + Math.PI / 6),
                        };

                        return (
                            <g key={`conn-${i}`}>
                                <line
                                    x1={startX} y1={startY}
                                    x2={endX} y2={endY}
                                    stroke={theme.colors.text.secondary}
                                    strokeWidth={1.5}
                                    opacity={0.6}
                                />
                                <polygon
                                    points={`${arrowTip.x},${arrowTip.y} ${arrowBase1.x},${arrowBase1.y} ${arrowBase2.x},${arrowBase2.y}`}
                                    fill={theme.colors.text.secondary}
                                    opacity={0.6}
                                />
                            </g>
                        );
                    })}

                    {/* Nodes */}
                    {previewNodes.map(node => {
                        const color = getNodeColor(node.type);
                        const label = getNodeLabel(node.type);

                        return (
                            <g key={node.id} transform={`translate(${node.position.x}, ${node.position.y})`}>
                                <circle
                                    r={nodeRadius}
                                    fill={color}
                                    stroke={theme.colors.border.primary}
                                    strokeWidth={1.5}
                                    opacity={0.9}
                                    filter="drop-shadow(0 1px 3px rgba(0,0,0,0.3))"
                                />
                                <text
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                    fill="white"
                                    fontSize={11}
                                    fontWeight={600}
                                    fontFamily={theme.typography.fontFamily}
                                    pointerEvents="none"
                                >
                                    {label}
                                </text>
                            </g>
                        );
                    })}
                </g>
            </svg>
        </div>
    );
};

