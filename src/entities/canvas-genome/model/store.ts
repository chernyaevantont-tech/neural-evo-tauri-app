import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { enableMapSet } from 'immer';

enableMapSet();
import { Connection, Position, VisualGenome, VisualNode } from './types';
import { Genome } from './genome';
import { v4 } from 'uuid';
import { BaseNode } from './nodes/base_node';
import { calculateLayoutForNewGraph } from '../lib/calculateLayout';

interface CanvasGenomeState {
    nodes: Map<string, VisualNode>;
    genomeNode: Map<string, VisualNode[]>;
    genomes: Map<string, VisualGenome>;
    connections: Map<string, Connection>;
    addNode: (node: BaseNode, postition: Position) => void;
    editNode: (nodeId: string, node: BaseNode) => void;
    deleteNode: (deletingNodeId: string) => void;
    moveNodes: (nodePositions: { nodeId: string, position: Position }[]) => void;
    highlightNodes: (nodesHighlight: { nodeId: string, isHighlighted: boolean }[]) => void;
    connectNodes: (fromNodeId: string, toNodeId: string) => void;
    deleteConnection: (deletingConnectitionId: string) => void;
    addGenome: (
        nodes: BaseNode[],
        genome: Genome,
        canvasWidth: number,
        canvasHeight: number,
        translateX: number,
        translateY: number,
        canvasScale: number,
        iterations: number,
    ) => void;
    deleteGenome: (genomeId: string) => void;
    reset: () => void;
}

export const useCanvasGenomeStore = create<CanvasGenomeState>()(
    immer((set) => ({
        nodes: new Map(),
        genomeNode: new Map(),
        genomes: new Map(),
        connections: new Map(),
        addNode: (node, position) =>
            set(state => {
                const newGenome = new Genome([node], [node])
                const newGenomeId = v4().toString();
                const newVisualNode = {
                    genomeId: newGenomeId,
                    node: node,
                    position: position,
                    highlighted: false
                };
                state.nodes.set(node.id, newVisualNode);
                state.genomes.set(newGenomeId, { id: newGenomeId, genome: newGenome, isValid: false });
                state.genomeNode.set(newGenomeId, [newVisualNode]);
            }),
        editNode: (nodeId, node) => {
            try {
                set(state => {
                    const oldVisualNode = state.nodes.get(nodeId);
                    if (oldVisualNode) {

                        const inConnections: string[] = [];
                        const outConnections: string[] = [];

                        state.connections.forEach((conn) => {
                            if (conn.toNodeId == nodeId) {
                                inConnections.push(conn.fromNodeId);
                                state.nodes.get(conn.fromNodeId)!.node.RemoveNext(oldVisualNode.node as BaseNode);
                            }
                            if (conn.fromNodeId == nodeId) {
                                outConnections.push(conn.toNodeId);
                                oldVisualNode.node.RemoveNext(state.nodes.get(conn.toNodeId)!.node as BaseNode);
                            }
                        });

                        const updatedVisualNode: VisualNode = {
                            ...oldVisualNode,
                            node: node,
                        };

                        state.nodes.delete(nodeId);
                        state.nodes.set(node.id, updatedVisualNode);

                        const genomeId = updatedVisualNode.genomeId;

                        state.genomeNode.set(
                            genomeId,
                            [...state.genomeNode.get(genomeId)!.filter(n => n.node.id != nodeId), updatedVisualNode]
                        );

                        const newConns = new Map<string, Connection>();

                        inConnections.forEach(fromId => {
                            const fromNode = state.nodes.get(fromId);
                            if (fromNode && fromNode.node.CheckCompability(node as BaseNode)) {
                                fromNode.node.AddNext(node);
                                const connId = v4();
                                newConns.set(connId, { id: connId, fromNodeId: fromId, toNodeId: node.id });
                            } else {
                                throw new Error();
                            }
                        });

                        outConnections.forEach(toId => {
                            const toNode = state.nodes.get(toId);
                            if (toNode && node.CheckCompability(toNode.node as BaseNode)) {
                                node.AddNext(toNode.node as BaseNode);
                                const connId = v4();
                                newConns.set(connId, { id: connId, fromNodeId: node.id, toNodeId: toId });
                            } else {
                                throw new Error();
                            }
                        });

                        state.connections.forEach((conn, connId) => {
                            if (conn.fromNodeId != nodeId && conn.toNodeId != nodeId) {
                                newConns.set(connId, conn);
                            }
                        });

                        state.connections = newConns;
                    }
                })
            } catch (e) {
                return e
            }
        },
        deleteNode: (deletingNodeId) =>
            set(state => {
                const deletingNode = state.nodes.get(deletingNodeId)!;
                if (state.genomeNode.get(deletingNode.genomeId)!.length > 1) {
                    for (const [id, conn] of state.connections) {
                        if (conn.fromNodeId === deletingNode.node.id || conn.toNodeId === deletingNode.node.id) {
                            state.connections.delete(id);
                        }
                    }
                    const connectedNodes = [...deletingNode.node.previous, ...deletingNode.node.next];
                    const checkedNodes = new Set<string>();
                    const nodeAsBase = deletingNode as unknown as VisualNode;
                    nodeAsBase.node.ClearAllConnections();

                    for (let node of connectedNodes) {
                        if (checkedNodes.has(node.id)) continue;
                        checkedNodes.add(node.id);

                        const newGenomeId = v4().toString();
                        const connectedVisualNodes: VisualNode[] = [];
                        const currentConnectedNodes = new Set<string>;
                        const nodesToCheck = [node];
                        const inputNodes: BaseNode[] = [];
                        const outputNodes: BaseNode[] = [];
                        let isValidFlag = true;

                        while (nodesToCheck.length > 0) {
                            const currentNode = nodesToCheck.shift()!;

                            if (!currentConnectedNodes.has(currentNode.id)) {
                                if (currentNode.previous.length == 0) {
                                    inputNodes.push(currentNode as BaseNode);
                                    if (currentNode.GetNodeType() != "Input") {
                                        isValidFlag = false;
                                    }
                                }
                                if (currentNode.next.length == 0) {
                                    outputNodes.push(currentNode as BaseNode);
                                    if (currentNode.GetNodeType() != "Output") {
                                        isValidFlag = false;
                                    }
                                }
                                const currentNodeId = currentNode.id;
                                const currentVisualNode = state.nodes.get(currentNodeId);
                                currentConnectedNodes.add(currentNode.id);
                                checkedNodes.add(currentNode.id);
                                if (currentVisualNode) {
                                    connectedVisualNodes.push({ ...(currentVisualNode as VisualNode), genomeId: newGenomeId });
                                    state.nodes.set(currentNodeId, { ...currentVisualNode, genomeId: newGenomeId });
                                }
                                nodesToCheck.push(...currentNode.previous, ...currentNode.next);
                            }
                        }

                        if (connectedVisualNodes.length == 0) continue;

                        const newGenome = new Genome(inputNodes, outputNodes);
                        state.genomeNode.set(newGenomeId, connectedVisualNodes);
                        state.genomes.set(newGenomeId, { id: newGenomeId, isValid: isValidFlag, genome: newGenome });
                    }
                }

                state.nodes.delete(deletingNode.node.id);
                state.genomeNode.delete(deletingNode.genomeId);
                state.genomes.delete(deletingNode.genomeId);
            }),
        moveNodes: (nodePositions: { nodeId: string, position: Position }[]) =>
            set(state => {
                nodePositions.forEach(nodePos => {
                    const currentNode = state.nodes.get(nodePos.nodeId);
                    if (currentNode) {
                        state.nodes.set(nodePos.nodeId, { ...currentNode, position: nodePos.position });
                    }
                });
            }),
        highlightNodes: (nodesHighlight: { nodeId: string, isHighlighted: boolean }[]) =>
            set(state => {
                nodesHighlight.forEach(nodeHighlight => {
                    const currentNode = state.nodes.get(nodeHighlight.nodeId);
                    if (currentNode) {
                        state.nodes.set(nodeHighlight.nodeId, { ...currentNode, highlighted: nodeHighlight.isHighlighted });
                    }
                })
            }),
        connectNodes: (fromNodeId, toNodeId) =>
            set(state => {
                const fromNode = state.nodes.get(fromNodeId);
                const toNode = state.nodes.get(toNodeId);

                if (fromNode && toNode) {

                    console.log("from node:\n", fromNode.node.GetInfo(), fromNode.node.GetInputShape());
                    console.log("to node:\n", toNode.node.GetInfo(), toNode.node.GetInputShape());

                    if (fromNode.node.CheckCompability(toNode.node as BaseNode)) {
                        console.log("a");

                        fromNode.node.AddNext(toNode.node as BaseNode);

                        const connectionId = v4().toString();
                        const connection: Connection = {
                            id: connectionId,
                            fromNodeId: fromNodeId,
                            toNodeId: toNodeId,
                        };

                        state.connections.set(connectionId, connection);

                        const fromGenomeId = fromNode.genomeId;
                        const toGenomeId = toNode.genomeId;

                        const updatedFromNode: VisualNode = { ...fromNode as VisualNode };
                        const updatedToNode: VisualNode = { ...toNode as VisualNode, genomeId: fromGenomeId };

                        state.nodes.set(fromNodeId, updatedFromNode);
                        state.nodes.set(toNodeId, updatedToNode);

                        const connectedNodesCheck = new Set<string>();
                        const nodesToCheck: BaseNode[] = [toNode.node as BaseNode];

                        while (nodesToCheck.length > 0) {
                            const currentNode = nodesToCheck.shift()!;
                            if (!connectedNodesCheck.has(currentNode.id)) {
                                connectedNodesCheck.add(currentNode.id);
                                const oldNode = state.nodes.get(currentNode.id)!;
                                state.nodes.set(currentNode.id, { ...oldNode, genomeId: fromGenomeId });
                                nodesToCheck.push(...currentNode.previous, ...currentNode.next);
                            }
                        }

                        const fromGenomeNode = state.genomeNode.get(fromGenomeId)!;
                        const fromGenome = state.genomes.get(fromGenomeId)!;

                        if (fromGenomeId != toGenomeId) {
                            const toGenomeNodes = state.genomeNode.get(toGenomeId)!;
                            toGenomeNodes.forEach(n => n.genomeId = fromGenomeId);

                            fromGenomeNode.push(...toGenomeNodes);

                            state.genomeNode.delete(toGenomeId);
                            state.genomes.delete(toGenomeId);
                        }

                        let isValidFlag = true;
                        const inputNodes: BaseNode[] = [];
                        const outputNodes: BaseNode[] = [];
                        for (let node of fromGenomeNode) {
                            if (node.node.previous.length == 0) {
                                inputNodes.push(node.node as BaseNode);
                                if (node.node.GetNodeType() != "Input") {
                                    isValidFlag = false;
                                }
                            }
                            if (node.node.next.length == 0) {
                                outputNodes.push(node.node as BaseNode);
                                if (node.node.GetNodeType() != "Output") {
                                    isValidFlag = false;
                                }
                            }
                        }

                        fromGenome.genome = new Genome(inputNodes, outputNodes);
                        fromGenome.isValid = isValidFlag;
                    }
                }
            }),
        deleteConnection: (deletingConnectionId) =>
            set(state => {
                const deletingConnection = state.connections.get(deletingConnectionId)!;

                const fromNode = state.nodes.get(deletingConnection.fromNodeId)!;
                const toNode = state.nodes.get(deletingConnection.toNodeId)!;

                const genomeId = toNode.genomeId;
                fromNode.node.RemoveNext(toNode.node as BaseNode);

                const toNodeConnectedNodes: VisualNode[] = [];
                const toConnectedNodesCheck = new Set<string>();
                const toNodesToCheck: BaseNode[] = [toNode.node as BaseNode];

                const toGenomeId = v4().toString();
                let toGenomeValidFlag = true;
                const toInputNodes: BaseNode[] = [];
                const toOutputNodes: BaseNode[] = [];

                let sameGenomeFlag = false;

                while (toNodesToCheck.length > 0) {
                    const currentNode = toNodesToCheck.shift()!;
                    if (!toConnectedNodesCheck.has(currentNode.id)) {
                        if (currentNode.id == fromNode.node.id && currentNode.id == toNode.node.id) {
                            sameGenomeFlag = true;
                            break;
                        }
                        if (currentNode.previous.length == 0) {
                            toInputNodes.push(currentNode);
                            if (currentNode.GetNodeType() != "Input") {
                                toGenomeValidFlag = false;
                            }
                        }
                        if (currentNode.next.length == 0) {
                            toOutputNodes.push(currentNode);
                            if (currentNode.GetNodeType() != "Output") {
                                toGenomeValidFlag = false;
                            }
                        }
                        toConnectedNodesCheck.add(currentNode.id);
                        toNodesToCheck.push(...currentNode.previous, ...currentNode.next);
                        const visualNode = state.nodes.get(currentNode.id);
                        if (visualNode) {
                            toNodeConnectedNodes.push({ ...visualNode, node: currentNode });
                            state.nodes.set(currentNode.id, { ...visualNode, genomeId: toGenomeId });
                        }
                    }
                }

                if (!sameGenomeFlag) {
                    const fromNodeConnectedNodes: VisualNode[] = [];
                    const fromConnectedNodesCheck = new Set<string>();
                    const fromNodesToCheck: BaseNode[] = [fromNode.node as BaseNode];

                    const fromGenomeId = v4().toString();
                    let fromGenomeValidFlag = true;
                    const fromInputNodes: BaseNode[] = [];
                    const fromOutputNodes: BaseNode[] = [];

                    while (fromNodesToCheck.length > 0) {
                        const currentNode = fromNodesToCheck.shift()!;
                        if (!fromConnectedNodesCheck.has(currentNode.id)) {
                            if (currentNode.previous.length == 0) {
                                fromInputNodes.push(currentNode);
                                if (currentNode.GetNodeType() != "Input") {
                                    fromGenomeValidFlag = false;
                                }
                            }
                            if (currentNode.next.length == 0) {
                                fromInputNodes.push(currentNode);
                                if (currentNode.GetNodeType() != "Output") {
                                    fromGenomeValidFlag = false;
                                }
                            }
                            fromConnectedNodesCheck.add(currentNode.id);
                            fromNodesToCheck.push(...currentNode.previous, ...currentNode.next);
                            const visualNode = state.nodes.get(currentNode.id);
                            if (visualNode) {
                                fromNodeConnectedNodes.push({ ...visualNode, node: currentNode });
                                state.nodes.set(currentNode.id, { ...visualNode, genomeId: fromGenomeId });
                            }
                        }
                    }

                    const toGenome = new Genome(toInputNodes, toOutputNodes);
                    const fromGenome = new Genome(fromInputNodes, fromOutputNodes);

                    state.genomes.delete(genomeId);
                    state.genomes.set(toGenomeId, { genome: toGenome, isValid: toGenomeValidFlag, id: toGenomeId });
                    state.genomes.set(fromGenomeId, { genome: fromGenome, isValid: fromGenomeValidFlag, id: fromGenomeId });

                    state.genomeNode.delete(genomeId);
                    state.genomeNode.set(toGenomeId, toNodeConnectedNodes);
                    state.genomeNode.set(fromGenomeId, fromNodeConnectedNodes);
                }

                state.connections.delete(deletingConnectionId);
            }),
        addGenome: (
            nodes,
            genome,
            canvasWidth,
            canvasHeight,
            translateX,
            translateY,
            canvasScale,
            iterations
        ) => set(state => {
            const newGenomeId = v4();
            const visualNodes: VisualNode[] = [];

            const newNodesPosition = calculateLayoutForNewGraph(
                nodes,
                canvasWidth,
                canvasHeight,
                translateX,
                translateY,
                canvasScale,
                iterations
            );

            nodes.forEach((node: BaseNode) => {
                const pos = newNodesPosition.get(node.id);
                const vNode: VisualNode = {
                    node,
                    position: pos,
                    genomeId: newGenomeId,
                    highlighted: false,
                }
                state.nodes.set(node.id, vNode);
                visualNodes.push(vNode);
            });

            let isValid = true;
            for (let node of nodes) {
                if (node.previous.length == 0 && node.GetNodeType() != "Input") {
                    isValid = false;
                }
                if (node.next.length == 0 && node.GetNodeType() != "Output") {
                    isValid = false;
                }
                node.next.forEach(nextNode => {
                    const connId = v4();
                    state.connections.set(connId, { id: connId, fromNodeId: node.id, toNodeId: nextNode.id })
                });
            }

            state.genomes.set(newGenomeId, { id: newGenomeId, genome, isValid });
            state.genomeNode.set(newGenomeId, visualNodes);
        }),
        deleteGenome: (genomeId: string) =>
            set(state => {
                state.genomes.delete(genomeId);
                const nodesToDelete = state.genomeNode.get(genomeId)!;
                const nodesToDeleteSet = new Set(nodesToDelete.map(n => n.node.id));
                state.genomeNode.delete(genomeId);
                nodesToDelete.forEach(node => {
                    state.nodes.delete(node.node.id);
                })
                for (const [id, conn] of state.connections) {
                    if (nodesToDeleteSet.has(conn.fromNodeId))
                        state.connections.delete(id);
                }
            }),
        reset: () => set(state => {
            state.nodes = new Map();
            state.genomeNode = new Map();
            state.genomes = new Map();
            state.connections = new Map();
        }),
    }))
)