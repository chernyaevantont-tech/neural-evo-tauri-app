import { useCallback, useRef, useState } from 'react';
import styles from './NetworkEditorPage.module.css';
import { VisualGenome, VisualNode } from '../../components/types';
import { NetworkCanvas, SidePanel, TitleBar } from '../../widgets';
import { SideMenu } from '../../widgets/side-menu/SideMenu';
import { useCanvasInteraction, useNetworkState } from './hooks';
import { BaseNode } from '../../evo/nodes/base_node';
import { Position } from '../../shared/types';
import { loadGenomeFromFile } from '../../shared/api';
import { v4 } from 'uuid';

export const NetworkEditorPage: React.FC = () => {
    const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
    const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
    
    const networkState = useNetworkState();
    const canvasState = useCanvasInteraction();

    const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
    const [configNodeType, setConfigNodeType] = useState<string | null>(null);
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    const openConfigPanel = useCallback((type: string, nodeId?: string) => {
        setConfigNodeType(type);
        setEditingNodeId(nodeId || null);
        setConfigPanelOpen(true);
    }, []);

    const handleGetSubgenome = () => {
        if (!networkState.selectedGenomeId) return;

        const newNodes = new Map(networkState.nodes);

        const currentGenomeNodes = networkState.genomeNode.get(networkState.selectedGenomeId)!;
        
        // Reset highlighting for all nodes in the genome
        for (let node of currentGenomeNodes) {
            const nodeId = node.node.id;
            const existingNode = newNodes.get(nodeId);
            if (existingNode) {
                newNodes.set(nodeId, { ...existingNode, highlighted: false });
            }
        }

        const subGenomeNodeIds = networkState.genomes.get(networkState.selectedGenomeId)!.genome.GetRandomSubgenome();
        for (let nodeId of subGenomeNodeIds) {
            const existingNode = newNodes.get(nodeId);
            if (existingNode) {
                newNodes.set(nodeId, { ...existingNode, highlighted: true });
            }
        }

        networkState.setNodes(newNodes);

        // Sync genomeNode state with new node objects
        const newGenomeNode = new Map(networkState.genomeNode);
        const updatedGenomeNodes = currentGenomeNodes
            .map(node => newNodes.get(node.node.id))
            .filter((n): n is VisualNode => n !== undefined);
            
        newGenomeNode.set(networkState.selectedGenomeId, updatedGenomeNodes);
        networkState.setGenomeNode(newGenomeNode);
    };

    const calculateLayoutForNewGraph = useCallback((
        nodesToLayout: BaseNode[],
        iterations: number = 300
    ): Map<string, Position> => {
        if (nodesToLayout.length === 0) return new Map();

        // Параметры физической симуляции
        const REPULSION_STRENGTH = 5000;
        const ATTRACTION_STRENGTH = 0.01;
        const DAMPING = 0.85;
        const MIN_DISTANCE = 50;
        const IDEAL_DISTANCE = 150;

        // Инициализация случайных позиций для новых нод
        const positions = new Map<string, Position>();
        const velocities = new Map<string, Position>();

        nodesToLayout.forEach((node, index) => {
            // Размещаем в круге для начального распределения
            const angle = (index / nodesToLayout.length) * Math.PI * 2;
            const radius = 100;
            positions.set(node.id, {
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius
            });
            velocities.set(node.id, { x: 0, y: 0 });
        });

        // Симуляция
        for (let iter = 0; iter < iterations; iter++) {
            const forces = new Map<string, Position>();

            nodesToLayout.forEach(node => {
                forces.set(node.id, { x: 0, y: 0 });
            });

            // Силы отталкивания между всеми парами
            for (let i = 0; i < nodesToLayout.length; i++) {
                for (let j = i + 1; j < nodesToLayout.length; j++) {
                    const node1 = nodesToLayout[i];
                    const node2 = nodesToLayout[j];
                    const pos1 = positions.get(node1.id)!;
                    const pos2 = positions.get(node2.id)!;

                    const dx = pos2.x - pos1.x;
                    const dy = pos2.y - pos1.y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                    if (distance < MIN_DISTANCE) continue;

                    const repulsionForce = REPULSION_STRENGTH / (distance * distance);
                    const fx = (dx / distance) * repulsionForce;
                    const fy = (dy / distance) * repulsionForce;

                    const force1 = forces.get(node1.id)!;
                    const force2 = forces.get(node2.id)!;
                    force1.x -= fx;
                    force1.y -= fy;
                    force2.x += fx;
                    force2.y += fy;
                }
            }

            // Силы притяжения для соединенных узлов
            nodesToLayout.forEach(node => {
                node.next.forEach(nextNode => {
                    const pos1 = positions.get(node.id);
                    const pos2 = positions.get(nextNode.id);

                    if (!pos1 || !pos2) return;

                    const dx = pos2.x - pos1.x;
                    const dy = pos2.y - pos1.y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                    const attractionForce = ATTRACTION_STRENGTH * (distance - IDEAL_DISTANCE);
                    const fx = (dx / distance) * attractionForce;
                    const fy = (dy / distance) * attractionForce;

                    const force1 = forces.get(node.id)!;
                    const force2 = forces.get(nextNode.id)!;
                    force1.x += fx;
                    force1.y += fy;
                    force2.x -= fx;
                    force2.y -= fy;
                });
            });

            // Применение сил
            nodesToLayout.forEach(node => {
                const velocity = velocities.get(node.id)!;
                const force = forces.get(node.id)!;
                const pos = positions.get(node.id)!;

                velocity.x = (velocity.x + force.x) * DAMPING;
                velocity.y = (velocity.y + force.y) * DAMPING;

                pos.x += velocity.x;
                pos.y += velocity.y;
            });
        }

        // Вычисление центра графа
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        positions.forEach(pos => {
            minX = Math.min(minX, pos.x);
            maxX = Math.max(maxX, pos.x);
            minY = Math.min(minY, pos.y);
            maxY = Math.max(maxY, pos.y);
        });

        const graphCenterX = (minX + maxX) / 2;
        const graphCenterY = (minY + maxY) / 2;

        // Центрирование относительно экрана
        if (svgRef.current) {
            const svg = svgRef.current;
            const rect = svg.getBoundingClientRect();
            const screenCenterX = rect.width / 2;
            const screenCenterY = rect.height / 2;
            const worldCenterX = (screenCenterX - canvasState.translate.x) / canvasState.scale;
            const worldCenterY = (screenCenterY - canvasState.translate.y) / canvasState.scale;

            const offsetX = worldCenterX - graphCenterX;
            const offsetY = worldCenterY - graphCenterY;

            // Применение смещения
            const finalPositions = new Map<string, Position>();
            positions.forEach((pos, id) => {
                finalPositions.set(id, {
                    x: Math.round(pos.x + offsetX),
                    y: Math.round(pos.y + offsetY)
                });
            });

            return finalPositions;
        }

        return positions;
    }, [canvasState.scale, canvasState.translate]);

    const handleLoadGenome = () => {
        loadGenomeFromFile((data: ReturnType<typeof import('../../saver/loadGenome').loadGenome>) => {
            const { nodes: loadedNodes, genome, isValid } = data;
            const loadedGenomeId = v4();
            const newNodes = new Map(networkState.nodes);
            const visualNodes: VisualNode[] = [];

            // Используем физическую симуляцию для размещения нод
            const newNodesPosition = calculateLayoutForNewGraph(loadedNodes);

            loadedNodes.forEach((node: BaseNode) => {
                const pos = newNodesPosition.get(node.id)!;
                const vNode: VisualNode = {
                    node,
                    position: pos,
                    genomeId: loadedGenomeId,
                    highlighted: false,
                };
                newNodes.set(node.id, vNode);
                visualNodes.push(vNode);
            });

            networkState.setNodes(newNodes);
            networkState.setGenomes((prev) => new Map(prev).set(loadedGenomeId, { id: loadedGenomeId, genome, isValid }));
            networkState.setGenomeNode((prev) => new Map(prev).set(loadedGenomeId, visualNodes));

            // Add connections from loaded genome
            networkState.setConnections((prev) => {
                const newConnections = new Map(prev);
                loadedNodes.forEach((node: BaseNode) => {
                    node.next.forEach((nextNode: BaseNode) => {
                        const connId = v4();
                        newConnections.set(connId, { id: connId, fromNodeId: node.id, toNodeId: nextNode.id });
                    });
                });
                return newConnections;
            });
        });
    };

    return (
        <div className={styles.container}>
            <TitleBar />

            <div className={styles.content}>
                <SideMenu
                    handleAddNode={openConfigPanel}
                    handleLoadGenome={handleLoadGenome}
                    handleGetSubgenome={handleGetSubgenome}
                />

                <NetworkCanvas
                    onNodeSelect={setSelectedNode}
                    onGenomeSelect={setSelectedGenome}
                    genomesState={[networkState.genomes, networkState.setGenomes]}
                    networkState={networkState}
                    canvasState={canvasState}
                    configPanelOpen={configPanelOpen}
                    setConfigPanelOpen={setConfigPanelOpen}
                    configNodeType={configNodeType}
                    setConfigNodeType={setConfigNodeType}
                    editingNodeId={editingNodeId}
                    setEditingNodeId={setEditingNodeId}
                    openConfigPanel={openConfigPanel}
                    svgRef={svgRef}
                />

                <SidePanel
                    selectedNode={selectedNode}
                    genomes={Array.from(networkState.genomes.values())}
                />
            </div>
        </div>
    );
}