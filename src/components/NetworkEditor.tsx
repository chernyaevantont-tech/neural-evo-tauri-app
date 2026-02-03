import React, { useState, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { VisualNode, Connection, Position, VisualGenome } from './types';
import { NodeRenderer } from './NodeRenderer';
import { ConnectionRenderer } from './ConnectionRenderer';
import { NodeConfigPanel } from './NodeConfigPanel';
import { BaseNode } from '../evo/nodes/base_node';
import { v4 } from "uuid"

import './NetworkEditor.css';
import { Genome } from '../evo/genome';
import { InputNode } from '../evo/nodes/layers/input_node';
import { OutputNode } from '../evo/nodes/layers/output_node';
import { loadGenomeApi } from '../api/genome/loadGenome';
import { ConnectionIndexes, loadGenome } from '../saver/loadGenome';
import { Subgenome } from '../evo/types';

interface NetworkEditorProps {
    onNodeSelect: (node: VisualNode | null) => void;
    onGenomeSelect: (genome: VisualGenome | null) => void;
    genomes: Map<string, VisualGenome>;
    setGenomes: React.Dispatch<React.SetStateAction<Map<string, VisualGenome>>>;
}

export const NetworkEditor: React.FC<NetworkEditorProps> = ({ onNodeSelect, onGenomeSelect, genomes, setGenomes }) => {
    const [nodes, setNodes] = useState<Map<string, VisualNode>>(new Map());
    const [genomeNode, setGenomeNode] = useState<Map<string, VisualNode[]>>(new Map);
    const [genomeSubgenomes, setGenomeSubgenomes] = useState<Map<string, Subgenome>>(new Map);
    const [connections, setConnections] = useState<Map<string, Connection>>(new Map());
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [selectedGenomeId, setSelectedGenomeId] = useState<string | null>(null);
    const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
    const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
    const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
    const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
    const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
    const [configNodeType, setConfigNodeType] = useState<string | null>(null);
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    const [nodeContextMenu, setNodeContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);
    const [connectionContextMenu, setConnectionContextMenu] = useState<{ x: number, y: number, connectionId: string } | null>(null);

    const [scale, setScale] = useState<number>(1);
    const [translate, setTranslate] = useState<Position>({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState<boolean>(false);
    const [panStart, setPanStart] = useState<Position>({ x: 0, y: 0 });

    const svgRef = useRef<SVGSVGElement>(null);
    const nodesRef = useRef<Map<string, VisualNode>>(nodes);
    const connectionsRef = useRef<Map<string, Connection>>(connections);

    React.useEffect(() => {
        nodesRef.current = nodes;
    }, [nodes]);

    React.useEffect(() => {
        connectionsRef.current = connections;
    }, [connections]);

    const openConfigPanel = useCallback((type: string, nodeId?: string) => {
        setConfigNodeType(type);
        setEditingNodeId(nodeId || null);
        setConfigPanelOpen(true);
    }, []);

    const handleConfigSave = useCallback((newNode: BaseNode) => {
        if (editingNodeId) {
            // Editing existing node - recreate with new parameters
            const oldVisualNode = nodesRef.current.get(editingNodeId);
            if (!oldVisualNode) return;

            // Store connections to restore
            const incomingConnections: string[] = [];
            const outgoingConnections: string[] = [];

            connectionsRef.current.forEach((conn) => {
                if (conn.toNodeId === editingNodeId) {
                    incomingConnections.push(conn.fromNodeId);
                }
                if (conn.fromNodeId === editingNodeId) {
                    outgoingConnections.push(conn.toNodeId);
                }
            });

            // Update node with new instance
            const updatedVisualNode: VisualNode = {
                ...oldVisualNode,
                node: newNode
            };

            // Replace the node ID in the map
            setNodes(prev => {
                const newNodes = new Map(prev);
                newNodes.set(editingNodeId, updatedVisualNode);
                return newNodes;
            });

            const oldNode = nodes.get(editingNodeId);

            setGenomeNode(prev => {
                const newGenomeNode = new Map(prev);
                newGenomeNode.set(selectedGenomeId!, [...prev.get(selectedGenomeId!)!.filter(n => n != oldNode), updatedVisualNode]);
                return newGenomeNode;
            });

            // Restore connections
            setConnections(prev => {
                const newConns = new Map<string, Connection>();

                // Re-add connections with the new node
                incomingConnections.forEach(fromId => {
                    const fromNode = nodesRef.current.get(fromId);
                    if (fromNode) {
                        // Check compatibility with new node
                        if (newNode.CheckCompability(fromNode.node)) {
                            fromNode.node.AddNext(newNode);

                            const connId = uuidv4();
                            newConns.set(connId, {
                                id: connId,
                                fromNodeId: fromId,
                                toNodeId: editingNodeId
                            });
                        }
                    }
                });

                outgoingConnections.forEach(toId => {
                    const toNode = nodesRef.current.get(toId);
                    if (toNode) {
                        // Check compatibility with new node
                        if (toNode.node.CheckCompability(newNode)) {
                            newNode.AddNext(toNode.node);
                            const connId = uuidv4();
                            newConns.set(connId, {
                                id: connId,
                                fromNodeId: editingNodeId,
                                toNodeId: toId
                            });
                        }
                    }
                });

                // Keep all other connections
                prev.forEach((conn, connId) => {
                    if (conn.fromNodeId !== editingNodeId && conn.toNodeId !== editingNodeId) {
                        newConns.set(connId, conn);
                    }
                });

                return newConns;
            });

            // Update selected node info if this node is currently selected
            if (selectedNodeId === editingNodeId && onNodeSelect) {
                onNodeSelect(updatedVisualNode);
            }
        } else {
            const newGenome: VisualGenome = {
                id: v4(),
                genome: new Genome([newNode], [newNode]),
                isValid: false,
            }

            const pos = { x: 100 + nodes.size * 20, y: 100 + nodes.size * 20 };

            const visualNode: VisualNode = {
                node: newNode,
                position: pos,
                genomeId: newGenome.id,
                highlighted: false,
            };

            setNodes(prev => new Map(prev).set(newNode.id, visualNode));

            setGenomes(prev => new Map(prev).set(newGenome.id, newGenome))
            setGenomeNode(prev => new Map(prev).set(newGenome.id, [visualNode]))
        }

        setConfigPanelOpen(false);
        setConfigNodeType(null);
        setEditingNodeId(null);
    }, [editingNodeId, configNodeType, selectedNodeId, onNodeSelect]);

    const handleConfigCancel = useCallback(() => {
        setConfigPanelOpen(false);
        setConfigNodeType(null);
        setEditingNodeId(null);
    }, []);

    const addNode = useCallback((type: string) => {
        openConfigPanel(type);
    }, [openConfigPanel]);

    const handleNodeDragStart = useCallback((nodeId: string, e: React.MouseEvent) => {
        e.preventDefault();
        const node = nodesRef.current.get(nodeId);

        if (!node || !svgRef.current) return;

        // Close context menu when starting to drag
        setNodeContextMenu(null);
        setConnectionContextMenu(null);

        const svg = svgRef.current;
        const rect = svg.getBoundingClientRect();

        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        setDragOffset({
            x: worldX - node.position.x,
            y: worldY - node.position.y
        });
        setDraggingNodeId(nodeId);
    }, [scale, translate]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (isPanning) {
            // Handle panning - adjust speed based on scale
            setNodeContextMenu(null);
            setConnectionContextMenu(null);
            const dx = e.clientX - panStart.x;
            const dy = e.clientY - panStart.y;
            setTranslate(prev => ({
                x: prev.x + dx,
                y: prev.y + dy
            }));
            setPanStart({ x: e.clientX, y: e.clientY });
        } else if (draggingNodeId && svgRef.current) {
            // Handle node dragging
            const svg = svgRef.current;
            const rect = svg.getBoundingClientRect();

            // Convert screen coordinates to world coordinates
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const worldX = (mouseX - translate.x) / scale;
            const worldY = (mouseY - translate.y) / scale;

            setNodes(prev => {
                const newNodes = new Map(prev);
                const node = newNodes.get(draggingNodeId);
                if (node) {
                    node.position = {
                        x: Math.round(worldX - dragOffset.x),
                        y: Math.round(worldY - dragOffset.y)
                    };
                    newNodes.set(draggingNodeId, { ...node });
                }
                return newNodes;
            });
        }
    }, [draggingNodeId, dragOffset, isPanning, panStart, scale, translate]);

    const handleMouseUp = useCallback((e: React.MouseEvent) => {
        if (e.button === 2) {
            // Right mouse button released
            setIsPanning(false);
        }
        setDraggingNodeId(null);
    }, []);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        if (e.button === 2) {
            // Right mouse button - start panning
            e.preventDefault();
            setIsPanning(true);
            setPanStart({ x: e.clientX, y: e.clientY });
        }
    }, []);

    const handleWheel = useCallback((e: React.WheelEvent) => {
        if (!svgRef.current) return;

        const svg = svgRef.current;
        const rect = svg.getBoundingClientRect();

        // Get mouse position relative to SVG
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Calculate mouse position in world coordinates (before zoom)
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        // Calculate new scale
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(5, scale * delta));

        // Calculate new translate to keep the point under mouse cursor
        const newTranslateX = mouseX - worldX * newScale;
        const newTranslateY = mouseY - worldY * newScale;

        setScale(newScale);
        setTranslate({ x: newTranslateX, y: newTranslateY });
    }, [scale, translate]);

    const handleNodeSelect = useCallback((nodeId: string) => {
        setSelectedNodeId(nodeId);
        setSelectedConnectionId(null);
        const node = nodesRef.current.get(nodeId);
        if (node && onNodeSelect) {
            setSelectedGenomeId(node.genomeId)
            onNodeSelect(node);
        }
    }, [onNodeSelect]);

    const handleConnectionSelect = (connectionId: string) => {
        setSelectedConnectionId(connectionId);
        setSelectedNodeId(null);
        setSelectedGenomeId(null);
    }

    const handleConnect = useCallback((nodeId: string) => {
        if (connectingFrom === null) {
            setConnectingFrom(nodeId);
        } else if (connectingFrom !== nodeId) {
            const fromNode = nodesRef.current.get(connectingFrom);
            const toNode = nodesRef.current.get(nodeId);

            if (fromNode && toNode) {
                // Check compatibility
                if (toNode.node.CheckCompability(fromNode.node)) {
                    fromNode.node.AddNext(toNode.node);

                    const connectionId = uuidv4();
                    const connection: Connection = {
                        id: connectionId,
                        fromNodeId: connectingFrom,
                        toNodeId: nodeId
                    };

                    setConnections(prev => new Map(prev).set(connectionId, connection));

                    const fromGenomeId = fromNode.genomeId;
                    const toGenomeId = toNode.genomeId;

                    // Create new VisualNode objects to force React update
                    const updatedFromNode: VisualNode = { ...fromNode };
                    const updatedToNode: VisualNode = { ...toNode, genomeId: fromGenomeId };

                    const newNodes = new Map(nodes);
                    newNodes.set(connectingFrom, updatedFromNode);
                    newNodes.set(nodeId, updatedToNode);

                    const connectedNodesMap: Map<BaseNode, boolean> = new Map();
                    const nodesToCheck: BaseNode[] = [toNode.node];

                    while (nodesToCheck.length > 0) {
                        const currentNode = nodesToCheck.shift()!;
                        if (!connectedNodesMap.get(currentNode)) {
                            connectedNodesMap.set(currentNode, true);
                            const oldNode = newNodes.get(currentNode.id)!;
                            newNodes.set(currentNode.id, { ...oldNode, genomeId: fromGenomeId });
                            nodesToCheck.push(...currentNode.previous, ...currentNode.next);
                        }
                    }

                    setNodes(newNodes);

                    const fromGenomeNode = genomeNode.get(fromGenomeId)!;
                    const fromGenome = genomes.get(fromGenomeId)!;

                    if (fromGenomeId != toGenomeId) {
                        const toGenomeNodes = genomeNode.get(toGenomeId)!;
                        toGenomeNodes.forEach(n => n.genomeId = fromGenomeId);

                        fromGenomeNode.push(...toGenomeNodes)

                        genomeNode.delete(toGenomeId);
                        genomes.delete(toGenomeId);
                    }

                    genomes.set(fromGenomeId, fromGenome);

                    let isValidFlag = true;
                    const inputNodes: BaseNode[] = [];
                    const outputNodes: BaseNode[] = [];
                    for (let node of fromGenomeNode) {
                        if (node.node.previous.length == 0) {
                            inputNodes.push(node.node);
                            if (!(node.node instanceof InputNode)) {
                                isValidFlag = false;
                            }
                        }
                        if (node.node.next.length == 0) {
                            outputNodes.push(node.node);
                            if (!(node.node instanceof OutputNode)) {
                                isValidFlag = false;
                            }
                        }
                    }

                    fromGenome.genome = new Genome(inputNodes, outputNodes);
                    fromGenome.isValid = isValidFlag;

                    setGenomes(genomes);


                    // Update selected node info if one of the connected nodes is selected
                    if (selectedNodeId === connectingFrom && onNodeSelect) {
                        onNodeSelect(updatedFromNode);
                    } else if (selectedNodeId === nodeId && onNodeSelect) {
                        onNodeSelect(updatedToNode);
                    }
                } else {
                    alert('Incompatible nodes! Cannot connect.');
                }
            }
            setConnectingFrom(null);
        }
    }, [connectingFrom, selectedNodeId, onNodeSelect]);


    const handleNodeContextMenu = useCallback((nodeId: string, e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setConnectionContextMenu(null);
        setNodeContextMenu({ x: e.clientX, y: e.clientY, nodeId });
    }, []);

    const handleNodeContextMenuEdit = useCallback(() => {
        if (!nodeContextMenu) return;
        const node = nodesRef.current.get(nodeContextMenu.nodeId);
        if (!node) return;

        setSelectedNodeId(nodeContextMenu.nodeId);
        openConfigPanel(node.node.GetNodeType(), nodeContextMenu.nodeId);
        setNodeContextMenu(null);
    }, [nodeContextMenu, openConfigPanel]);

    const handleNodeContextMenuCopy = useCallback(() => {
        if (!nodeContextMenu) return;
        const node = nodesRef.current.get(nodeContextMenu.nodeId);
        if (!node) return;

        const newNodes = new Map(nodes);
        const copyNode = node.node.Clone();
        const newGenome

        newNodes.set(copyNode.id, {})

        set
        setNodeContextMenu(null);
    }, [nodeContextMenu, openConfigPanel]);

    const handleNodeContextMenuDelete = useCallback(() => {
        if (!nodeContextMenu) return;

        const nodeIdToDelete = nodeContextMenu.nodeId;
        const nodeToDelete = nodesRef.current.get(nodeIdToDelete);
        if (!nodeToDelete) return;

        const deletingNode = nodeToDelete;
        setNodeContextMenu(null);

        // Remove connections
        setConnections(prev => {
            const newConns = new Map(prev);
            for (const [id, conn] of newConns) {
                if (conn.fromNodeId === nodeIdToDelete || conn.toNodeId === nodeIdToDelete) {
                    newConns.delete(id);
                }
            }
            return newConns;
        });

        const newNodes = new Map(nodesRef.current);
        newNodes.delete(nodeIdToDelete);

        const nodeGenomeId = deletingNode.genomeId;

        if (genomeNode.get(nodeGenomeId)!.length > 1) {
            const newGenomes = new Map(genomes);
            const newGenomeNode = new Map(genomeNode);

            const connectedNodes = [...deletingNode.node.previous, ...deletingNode.node.next];
            const checkedNodes = new Map<BaseNode, boolean>(connectedNodes.map(n => [n, false]));

            deletingNode.node.ClearAllConnections();

            for (let node of connectedNodes) {
                if (checkedNodes.get(node)) continue;
                checkedNodes.set(node, true);

                const newGenomeId = v4();
                const connectedVisualNodes: VisualNode[] = [];
                const connectedNodesMap: Map<BaseNode, boolean> = new Map();
                const nodesToCheck: BaseNode[] = [node];
                const inputNodes: BaseNode[] = [];
                const outputNodes: BaseNode[] = [];
                let isValidFlag = true;

                while (nodesToCheck.length > 0) {
                    const currentNode = nodesToCheck.shift()!;

                    if (!connectedNodesMap.get(currentNode)) {
                        if (currentNode.previous.length == 0) {
                            inputNodes.push(currentNode);
                            if (!(currentNode instanceof InputNode)) {
                                isValidFlag = false;
                            }
                        }
                        if (currentNode.next.length == 0) {
                            outputNodes.push(currentNode);
                            if (!(currentNode instanceof OutputNode)) {
                                isValidFlag = false;
                            }
                        }
                        const currentNodeId = currentNode.id;
                        const currentVisualNode = nodesRef.current.get(currentNodeId)!;
                        connectedNodesMap.set(currentNode, true);
                        connectedVisualNodes.push({ ...nodesRef.current.get(currentNodeId)!, genomeId: newGenomeId });
                        newNodes.set(currentNodeId, { ...currentVisualNode, genomeId: newGenomeId });
                        nodesToCheck.push(...currentNode.previous, ...currentNode.next);
                    }
                }

                if (connectedVisualNodes.length == 0) continue;

                const newGenome = new Genome(inputNodes, outputNodes);
                newGenomeNode.set(newGenomeId, connectedVisualNodes);
                newGenomes.set(newGenomeId, { id: newGenomeId, isValid: isValidFlag, genome: newGenome });
            }

            newGenomes.delete(nodeGenomeId);
            newGenomeNode.delete(nodeGenomeId);
            setGenomeNode(newGenomeNode);
            setGenomes(newGenomes);
        } else {
            setGenomeNode(prev => {
                const newGenomeNode = new Map(prev);
                newGenomeNode.delete(nodeGenomeId);
                return newGenomeNode;
            });
            setGenomes(prev => {
                const newGenomes = new Map(prev);
                newGenomes.delete(nodeGenomeId);
                return newGenomes;
            });
        }

        setNodes(newNodes);
        setSelectedNodeId(null);
        setSelectedConnectionId(null);
        setSelectedGenomeId(null);
        if (onNodeSelect) {
            onNodeSelect(null);
        }
    }, [nodeContextMenu, genomeNode, genomes, onNodeSelect]);

    const handleConnectionContextMenu = useCallback((connectionId: string, e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setNodeContextMenu(null);
        setConnectionContextMenu({ x: e.clientX, y: e.clientY, connectionId: connectionId });
    }, []);

    const handleConnectionContextMenuDelete = useCallback(() => {
        if (!connectionContextMenu) return;

        const connectionId = connectionContextMenu.connectionId;
        const connectionToDelete = connectionsRef.current.get(connectionId);
        if (!connectionToDelete) return;

        setConnectionContextMenu(null);

        setConnections(prev => {
            const newConnections = new Map(prev);
            newConnections.delete(connectionId);
            return newConnections;
        })

        const newNodes = new Map(nodesRef.current);
        const newGenomes = new Map(genomes);
        const newGenomeNode = new Map(genomeNode);

        const fromNode = newNodes.get(connectionToDelete.fromNodeId)!;
        const toNode = newNodes.get(connectionToDelete.toNodeId)!;

        const genomeId = toNode.genomeId;

        fromNode.node.RemoveNext(toNode.node);

        const toNodeConnectedNodes: VisualNode[] = [];
        const toConnectedNodesMap: Map<BaseNode, boolean> = new Map();
        const toNodesToCheck: BaseNode[] = [toNode.node];

        const toGenomeId = v4();
        let toGenomeValidFlag = true;
        const toInputNodes: BaseNode[] = [];
        const toOutputNodes: BaseNode[] = [];

        let sameGenomeFlag = false;

        while (toNodesToCheck.length > 0) {
            const currentNode = toNodesToCheck.shift()!;
            if (!toConnectedNodesMap.get(currentNode)) {
                if (currentNode.id == fromNode.node.id && currentNode.id == toNode.node.id) {
                    sameGenomeFlag = true;
                    break;
                }
                if (currentNode.previous.length == 0) {
                    toInputNodes.push(currentNode);
                    if (!(currentNode instanceof InputNode)) {
                        toGenomeValidFlag = false;
                    }
                }
                if (currentNode.next.length == 0) {
                    toOutputNodes.push(currentNode);
                    if (!(currentNode instanceof OutputNode)) {
                        toGenomeValidFlag = false;
                    }
                }
                toConnectedNodesMap.set(currentNode, true);
                toNodesToCheck.push(...currentNode.previous, ...currentNode.next);
                toNodeConnectedNodes.push({ ...nodes.get(currentNode.id)!, node: currentNode });
                newNodes.set(currentNode.id, { ...nodes.get(currentNode.id)!, genomeId: toGenomeId });
            }
        }

        if (!sameGenomeFlag) {
            const fromNodeConnectedNodes: VisualNode[] = [];
            const fromConnectedNodesMap: Map<BaseNode, boolean> = new Map();
            const fromNodesToCheck: BaseNode[] = [fromNode.node];

            const fromGenomeId = v4();
            let fromGenomeValidFlag = true;
            const fromInputNodes: BaseNode[] = [];
            const fromOutputNodes: BaseNode[] = [];

            while (fromNodesToCheck.length > 0) {
                const currentNode = fromNodesToCheck.shift()!;
                if (!fromConnectedNodesMap.get(currentNode)) {
                    if (currentNode.previous.length == 0) {
                        fromInputNodes.push(currentNode);
                        if (!(currentNode instanceof InputNode)) {
                            fromGenomeValidFlag = false;
                        }
                    }
                    if (currentNode.next.length == 0) {
                        fromOutputNodes.push(currentNode);
                        if (!(currentNode instanceof OutputNode)) {
                            fromGenomeValidFlag = false;
                        }
                    }
                    fromConnectedNodesMap.set(currentNode, true);
                    fromNodesToCheck.push(...currentNode.previous, ...currentNode.next);
                    fromNodeConnectedNodes.push({ ...nodes.get(currentNode.id)!, node: currentNode });
                    newNodes.set(currentNode.id, { ...nodes.get(currentNode.id)!, genomeId: fromGenomeId });
                }
            }

            const toGenome = new Genome(toInputNodes, toOutputNodes);
            const fromGenome = new Genome(fromInputNodes, fromOutputNodes);

            newGenomes.delete(genomeId);
            newGenomes.set(toGenomeId, { genome: toGenome, isValid: toGenomeValidFlag, id: toGenomeId });
            newGenomes.set(fromGenomeId, { genome: fromGenome, isValid: fromGenomeValidFlag, id: fromGenomeId });
            setGenomes(newGenomes);

            newGenomeNode.delete(genomeId);
            newGenomeNode.set(toGenomeId, toNodeConnectedNodes);
            newGenomeNode.set(fromGenomeId, fromNodeConnectedNodes);
            setGenomeNode(newGenomeNode);

            setNodes(newNodes);
        }
    }, [connectionContextMenu, genomeNode, genomes]);

    // Функция 1: Расчет позиций для нового загруженного графа с центрированием
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
            const worldCenterX = (screenCenterX - translate.x) / scale;
            const worldCenterY = (screenCenterY - translate.y) / scale;

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
    }, [scale, translate]);

    // Функция 2: Глобальная симуляция для всех существующих нод
    const applyGlobalForceDirectedLayout = useCallback((iterations: number = 200) => {
        if (nodes.size === 0) return;

        // Параметры (более слабые для сохранения структуры)
        const REPULSION_STRENGTH = 3000;
        const ATTRACTION_STRENGTH = 0.015;
        const DAMPING = 0.9;
        const MIN_DISTANCE = 50;
        const IDEAL_DISTANCE = 150;

        // Используем текущие позиции как начальные
        const positions = new Map<string, Position>();
        const velocities = new Map<string, Position>();

        nodes.forEach((node, id) => {
            positions.set(id, { ...node.position });
            velocities.set(id, { x: 0, y: 0 });
        });

        const nodeIds = Array.from(nodes.keys());

        // Симуляция
        for (let iter = 0; iter < iterations; iter++) {
            const forces = new Map<string, Position>();

            nodeIds.forEach(id => {
                forces.set(id, { x: 0, y: 0 });
            });

            // Силы отталкивания между всеми парами
            for (let i = 0; i < nodeIds.length; i++) {
                for (let j = i + 1; j < nodeIds.length; j++) {
                    const id1 = nodeIds[i];
                    const id2 = nodeIds[j];
                    const pos1 = positions.get(id1)!;
                    const pos2 = positions.get(id2)!;

                    const dx = pos2.x - pos1.x;
                    const dy = pos2.y - pos1.y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                    if (distance < MIN_DISTANCE) continue;

                    const repulsionForce = REPULSION_STRENGTH / (distance * distance);
                    const fx = (dx / distance) * repulsionForce;
                    const fy = (dy / distance) * repulsionForce;

                    const force1 = forces.get(id1)!;
                    const force2 = forces.get(id2)!;
                    force1.x -= fx;
                    force1.y -= fy;
                    force2.x += fx;
                    force2.y += fy;
                }
            }

            // Силы притяжения для соединений
            connections.forEach(conn => {
                const pos1 = positions.get(conn.fromNodeId);
                const pos2 = positions.get(conn.toNodeId);

                if (!pos1 || !pos2) return;

                const dx = pos2.x - pos1.x;
                const dy = pos2.y - pos1.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                const attractionForce = ATTRACTION_STRENGTH * (distance - IDEAL_DISTANCE);
                const fx = (dx / distance) * attractionForce;
                const fy = (dy / distance) * attractionForce;

                const force1 = forces.get(conn.fromNodeId)!;
                const force2 = forces.get(conn.toNodeId)!;
                force1.x += fx;
                force1.y += fy;
                force2.x -= fx;
                force2.y -= fy;
            });

            // Применение сил
            nodeIds.forEach(id => {
                const velocity = velocities.get(id)!;
                const force = forces.get(id)!;
                const pos = positions.get(id)!;

                velocity.x = (velocity.x + force.x) * DAMPING;
                velocity.y = (velocity.y + force.y) * DAMPING;

                pos.x += velocity.x;
                pos.y += velocity.y;
            });
        }

        // Применение новых позиций (без центрирования)
        setNodes(prev => {
            const newNodes = new Map(prev);
            positions.forEach((pos, id) => {
                const node = newNodes.get(id);
                if (node) {
                    node.position = {
                        x: Math.round(pos.x),
                        y: Math.round(pos.y)
                    };
                    newNodes.set(id, { ...node });
                }
            });
            return newNodes;
        });
    }, [nodes, connections]);

    const handleGenomeLoad = (
        loadedNodes: BaseNode[],
        loadedGenome: Genome,
        connectionIndexes: ConnectionIndexes,
        isValid: boolean,
    ) => {
        const newNodes = new Map(nodes);
        const newGenomes = new Map(genomes);
        const newGenomeNode = new Map(genomeNode);
        const newConnections = new Map(connections);

        const newNodesPosition = calculateLayoutForNewGraph(loadedNodes);
        const loadedGenomeId = v4();

        const visualNodes: VisualNode[] = [];

        for (let node of loadedNodes) {
            const newVisualNode: VisualNode = { node: node, position: newNodesPosition.get(node.id)!, genomeId: loadedGenomeId, highlighted: false };
            newNodes.set(node.id, newVisualNode);
            visualNodes.push(newVisualNode);
        }

        newGenomes.set(loadedGenomeId, { id: loadedGenomeId, genome: loadedGenome, isValid: isValid });
        newGenomeNode.set(loadedGenomeId, visualNodes);

        for (let connectionUnit of connectionIndexes) {
            const newConnectionId = v4();
            const fromNodeId = loadedNodes[connectionUnit.fromIndex].id;
            const toNodeId = loadedNodes[connectionUnit.toIndex].id;
            newConnections.set(newConnectionId, { id: newConnectionId, fromNodeId: fromNodeId, toNodeId: toNodeId });
        }

        setNodes(newNodes);
        setGenomes(newGenomes);
        setGenomeNode(newGenomeNode);
        setConnections(newConnections);
    }

    const handleGetRandomSubgraph = () => {
        if (!selectedGenomeId) return;

        const newNodes = new Map(nodes);
        
        const currentGenomeNodes = genomeNode.get(selectedGenomeId)!;
        for (let node of currentGenomeNodes) {
            node.highlighted = false;
        }

        const subGenomeNodeIds = genomes.get(selectedGenomeId)!.genome.GetRandomSubgenome();
        for (let nodeId of subGenomeNodeIds) {
            newNodes.get(nodeId)!.highlighted = true;
        }

        setNodes(newNodes);
    }

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>

            {/* Config Panel Modal */}
            {configPanelOpen && configNodeType && (
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000
                }}>
                    <div style={{
                        backgroundColor: 'white',
                        padding: '20px',
                        borderRadius: '8px',
                        maxWidth: '500px',
                        maxHeight: '80vh',
                        overflow: 'auto',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.3)'
                    }}>
                        <NodeConfigPanel
                            nodeType={configNodeType}
                            existingNode={editingNodeId ? nodes.get(editingNodeId)?.node : undefined}
                            onSave={handleConfigSave}
                            onCancel={handleConfigCancel}
                        />
                    </div>
                </div>
            )}

            <div style={{
                position: 'absolute',
                top: 10,
                left: 10,
                zIndex: 10,
                background: 'white',
                padding: '10px',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
            }}>D
                <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', maxWidth: '200px' }}>
                    <button onClick={() => addNode('Input')} style={buttonStyle}>+ Input</button>
                    <button onClick={() => addNode('Dense')} style={buttonStyle}>+ Dense</button>
                    <button onClick={() => addNode('Conv2D')} style={buttonStyle}>+ Conv2D</button>
                    <button onClick={() => addNode('Pooling')} style={buttonStyle}>+ Pooling</button>
                    <button onClick={() => addNode("Flatten")} style={buttonStyle}>+ Flatten</button>
                    <button onClick={() => addNode('Add')} style={buttonStyle}>+ Add</button>
                    <button onClick={() => addNode('Concat2D')} style={buttonStyle}>+ Concat</button>
                    <button onClick={() => addNode('Output')} style={buttonStyle}>+ Output</button>
                </div>
                <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', maxWidth: '200px' }}>
                    <button style={buttonStyle}
                        onClick={() => loadGenomeApi((genomeStr) => {
                            const { nodes, genome, connectionIndexes, isValid } = loadGenome(genomeStr);
                            handleGenomeLoad(nodes, genome, connectionIndexes, isValid);
                        })}>Load</button>
                </div>
                <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', maxWidth: '200px' }}>
                    <button style={buttonStyle}
                    onClick={handleGetRandomSubgraph}>Get subgenome</button>
                </div>
            </div>

            {/* Context Menu */}
            {nodeContextMenu && (
                <div
                    style={{
                        position: 'fixed',
                        left: nodeContextMenu.x,
                        top: nodeContextMenu.y,
                        backgroundColor: 'white',
                        border: '1px solid #ccc',
                        zIndex: 2000,
                        minWidth: '120px',
                    }}
                    onMouseDown={(e) => e.stopPropagation()}
                >
                    <button
                        onClick={handleNodeContextMenuEdit}
                        style={{
                            ...nodeContextMenuItemStyle,
                            borderBottom: '1px solid #eee'
                        }}
                    >
                        ✏️ Edit Node
                    </button>
                    <button
                        onClick={handleNodeContextMenuDelete}
                        style={{
                            ...nodeContextMenuItemStyle,
                            color: '#f44336'
                        }}
                    >
                        🗑️ Delete Node
                    </button>
                </div>
            )}
            {connectionContextMenu && (
                <div
                    style={{
                        position: 'fixed',
                        left: connectionContextMenu.x,
                        top: connectionContextMenu.y,
                        backgroundColor: 'white',
                        border: '1px solid #ccc',
                        zIndex: 2000,
                        minWidth: '120px',
                    }}
                    onMouseDown={(e) => e.stopPropagation()}
                >
                    <button
                        onClick={handleConnectionContextMenuDelete}
                        style={{
                            ...nodeContextMenuItemStyle,
                            color: '#f44336'
                        }}
                    >
                        🗑️ Delete Connection
                    </button>
                </div>
            )}

            <svg
                ref={svgRef}
                width="100%"
                height="100%"
                style={{ background: '#f5f5f5', cursor: isPanning ? 'grabbing' : 'default' }}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseDown={handleMouseDown}
                onWheel={handleWheel}
                onContextMenu={(e) => e.preventDefault()}
                onClick={(e) => {
                    if (e.target === svgRef.current) {
                        setSelectedNodeId(null);
                        setSelectedGenomeId(null);
                        setSelectedConnectionId(null);
                        setConnectingFrom(null);
                        setNodeContextMenu(null);
                        setConnectionContextMenu(null);
                        setConnectionContextMenu(null);
                        if (onNodeSelect) onNodeSelect(null);
                        if (onGenomeSelect) onGenomeSelect(null);
                    } else {
                        setNodeContextMenu(null);
                        setConnectionContextMenu(null);
                    }
                }}
            >
                <g transform={`translate(${translate.x}, ${translate.y}) scale(${scale})`}>
                    {Array.from(connections.values()).map(conn => (
                        <ConnectionRenderer
                            key={conn.id}
                            connection={conn}
                            nodes={nodes}
                            isSelected={selectedConnectionId == conn.id}
                            onSelect={handleConnectionSelect}
                            onContextMenu={handleConnectionContextMenu}
                        />
                    ))}

                    {Array.from(nodes.values()).map(node => (
                        <g
                            key={node.node.id}
                            onClickCapture={(e) => {
                                if (e.shiftKey) {
                                    e.stopPropagation();
                                    handleConnect(node.node.id);
                                } else if (connectingFrom !== null) {
                                    setConnectingFrom(null);
                                }
                            }}
                        >
                            <NodeRenderer
                                node={node}
                                isSelected={selectedNodeId === node.node.id}
                                onSelect={handleNodeSelect}
                                onDragStart={handleNodeDragStart}
                                onContextMenu={handleNodeContextMenu}
                            />
                        </g>
                    ))}
                </g>
            </svg>
        </div>
    );
};

const buttonStyle: React.CSSProperties = {
    padding: '5px 10px',
    fontSize: '12px',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: '4px',
    backgroundColor: '#2196F3',
    color: 'white',
    whiteSpace: 'nowrap'
};

const nodeContextMenuItemStyle: React.CSSProperties = {
    display: 'block',
    width: '100%',
    padding: '8px 12px',
    fontSize: '13px',
    textAlign: 'left',
    border: 'none',
    borderRadius: '0',
    background: 'white',
    cursor: 'pointer',
    color: '#333'
};
