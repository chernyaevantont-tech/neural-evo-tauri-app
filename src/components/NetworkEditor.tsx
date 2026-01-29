import React, { useState, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { VisualNode, Connection, Position, NodeType, VisualGenome } from './types';
import { NodeRenderer } from './NodeRenderer';
import { ConnectionRenderer } from './ConnectionRenderer';
import { NodeConfigPanel } from './NodeConfigPanel';
import { BaseNode } from '../evo/nodes/base_node';
import './NetworkEditor.css';

interface NetworkEditorProps {
    onNodeSelect?: (node: VisualNode | null) => void;
    onGenomeSelect?: (genome: VisualGenome | null) => void;
}

export const NetworkEditor: React.FC<NetworkEditorProps> = ({ onNodeSelect, onGenomeSelect }) => {
    const [nodes, setNodes] = useState<Map<string, VisualNode>>(new Map());
    const [genomes, setGenomes] = useState<Map<string, VisualGenome>>(new Map());
    const [connections, setConnections] = useState<Map<string, Connection>>(new Map());
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [selectedGenomeId, setSelectedGenomeId] = useState<string | null>(null); 
    const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
    const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
    const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
    const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
    const [configNodeType, setConfigNodeType] = useState<NodeType | null>(null);
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    
    const [scale, setScale] = useState<number>(1);
    const [translate, setTranslate] = useState<Position>({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState<boolean>(false);
    const [panStart, setPanStart] = useState<Position>({ x: 0, y: 0 });
    
    const svgRef = useRef<SVGSVGElement>(null);
    const nodesRef = useRef<Map<string, VisualNode>>(nodes);
    const genomesRef = useRef<Map<string, VisualGenome>>(genomes);
    const connectionsRef = useRef<Map<string, Connection>>(connections);

    React.useEffect(() => {
        nodesRef.current = nodes;
    }, [nodes]);

    React.useEffect(() => {
        connectionsRef.current = connections;
    }, [connections]);

    const openConfigPanel = useCallback((type: NodeType, nodeId?: string) => {
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
            // Creating new node
            setNodes(prev => {
                const pos = { x: 100 + prev.size * 20, y: 100 + prev.size * 20 };
                const id = newNode.id as string;

                const visualNode: VisualNode = {
                    id,
                    node: newNode,
                    position: pos,
                    type: configNodeType!,
                    genomeId: null,
                };

                return new Map(prev).set(id, visualNode);
            });
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

    const addNode = useCallback((type: NodeType) => {
        openConfigPanel(type);
    }, [openConfigPanel]);

    const handleNodeDragStart = useCallback((nodeId: string, e: React.MouseEvent) => {
        e.preventDefault();
        const node = nodesRef.current.get(nodeId);
        if (!node || !svgRef.current) return;

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
        e.preventDefault();
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
        const node = nodesRef.current.get(nodeId);
        if (node && onNodeSelect) {
            setSelectedGenomeId(node.genomeId)
            onNodeSelect(node);
        }
    }, [onNodeSelect]);

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

                    // Create new VisualNode objects to force React update
                    const updatedFromNode: VisualNode = { ...fromNode };
                    const updatedToNode: VisualNode = { ...toNode };

                    // Update nodes map with new objects
                    setNodes(prev => {
                        const newNodes = new Map(prev);
                        newNodes.set(connectingFrom, updatedFromNode);
                        newNodes.set(nodeId, updatedToNode);
                        return newNodes;
                    });

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

    const deleteSelectedNode = useCallback(() => {
        if (!selectedNodeId) return;

        // Remove connections
        setConnections(prev => {
            const newConns = new Map(prev);
            for (const [id, conn] of newConns) {
                if (conn.fromNodeId === selectedNodeId || conn.toNodeId === selectedNodeId) {
                    newConns.delete(id);
                }
            }
            return newConns;
        });

        // Remove node
        setNodes(prev => {
            const newNodes = new Map(prev);
            newNodes.delete(selectedNodeId);
            return newNodes;
        });

        setSelectedNodeId(null);
        if (onNodeSelect) {
            onNodeSelect(null);
        }
    }, [selectedNodeId, onNodeSelect]);

    const editSelectedNode = useCallback(() => {
        if (!selectedNodeId) return;
        const node = nodesRef.current.get(selectedNodeId);
        if (!node) return;

        openConfigPanel(node.type, selectedNodeId);
    }, [selectedNodeId, openConfigPanel]);

    // const getRandomSubgraph = useCallback(() => {
         //TODO
    // }, [selectedGraphId])

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
            }}>
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
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #ccc' }}>
                    <button 
                        onClick={editSelectedNode} 
                        disabled={!selectedNodeId}
                        style={{ ...buttonStyle, backgroundColor: '#FF9800', width: '100%', marginBottom: '5px' }}
                    >
                        Edit Node
                    </button>
                    <button 
                        onClick={deleteSelectedNode} 
                        disabled={!selectedNodeId}
                        style={{ ...buttonStyle, backgroundColor: '#f44336', width: '100%' }}
                    >
                        Delete Node
                    </button>

                </div>
            </div>

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
                        setConnectingFrom(null);
                        if (onNodeSelect) onNodeSelect(null);
                    }
                }}
            >
                <g transform={`translate(${translate.x}, ${translate.y}) scale(${scale})`}>
                {Array.from(connections.values()).map(conn => (
                    <ConnectionRenderer
                        key={conn.id}
                        connection={conn}
                        nodes={nodes}
                    />
                ))}

                {Array.from(nodes.values()).map(node => (
                    <g
                        key={node.id}
                        onClickCapture={(e) => {
                            if (e.shiftKey) {
                                e.stopPropagation();
                                handleConnect(node.id);
                            }
                        }}
                    >
                        <NodeRenderer
                            node={node}
                            isSelected={selectedNodeId === node.id}
                            onSelect={handleNodeSelect}
                            onDragStart={handleNodeDragStart}
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
