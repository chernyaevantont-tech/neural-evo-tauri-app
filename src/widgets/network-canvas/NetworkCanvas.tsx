import React, { useRef, useCallback, CSSProperties, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { VisualNode, Connection, VisualGenome, Position } from '../../shared/types';
import { NodeCard } from '../../entities/node/ui';
import { ConnectionLine } from '../../entities/connection/ui';
import { BaseNode } from '../../evo/nodes/base_node';
import { Genome } from '../../evo/genome';
import { InputNode } from '../../evo/nodes/layers/input_node';
import { OutputNode } from '../../evo/nodes/layers/output_node';
import { theme } from '../../shared/lib';
import { loadGenomeFromFile } from '../../shared/api';
import { useNetworkState, useCanvasInteraction, createNewGenomeWithNode, updateGenomeValidity } from './hooks';
import { NodeConfigForm, NodeToolbar } from '../../features/node-toolbar';
import { ContextMenu } from '../../features/genome-operations';

interface NetworkCanvasProps {
  onNodeSelect: (node: VisualNode | null) => void;
  onGenomeSelect: (genome: VisualGenome | null) => void;
  genomesState: [Map<string, VisualGenome>, React.Dispatch<React.SetStateAction<Map<string, VisualGenome>>>];
}

export const NetworkCanvas: React.FC<NetworkCanvasProps> = ({
  onNodeSelect,
  onGenomeSelect,
  genomesState,
}) => {
  const [genomes, setGenomes] = genomesState;
  
  const networkState = useNetworkState();
  const {
    nodes,
    setNodes,
    genomeNode,
    setGenomeNode,
    connections,
    setConnections,
    selectedNodeId,
    setSelectedNodeId,
    selectedGenomeId,
    setSelectedGenomeId,
    selectedConnectionId,
    setSelectedConnectionId,
  } = networkState;

  const canvasState = useCanvasInteraction();
  const {
    draggingNodeId,
    setDraggingNodeId,
    connectingFrom,
    setConnectingFrom,
    dragOffset,
    setDragOffset,
    scale,
    translate,
    setTranslate,
    isPanning,
    setIsPanning,
    panStart,
    setPanStart,
    handleWheel,
  } = canvasState;

  const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
  const [configNodeType, setConfigNodeType] = useState<string | null>(null);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [nodeContextMenu, setNodeContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);
  const [connectionContextMenu, setConnectionContextMenu] = useState<{ x: number; y: number; connectionId: string } | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const nodesRef = useRef<Map<string, VisualNode>>(nodes);
  const connectionsRef = useRef<Map<string, Connection>>(connections);
  const panningRef = useRef<boolean>(false);
  const lastMousePosRef = useRef<Position>({ x: 0, y: 0 });

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
      // Edit existing node logic (preserving connections)
      const oldVisualNode = nodesRef.current.get(editingNodeId);
      if (!oldVisualNode) return;

      const incomingConnections: string[] = [];
      const outgoingConnections: string[] = [];

      connectionsRef.current.forEach((conn) => {
        if (conn.toNodeId === editingNodeId) {
          incomingConnections.push(conn.fromNodeId);
          nodesRef.current.get(conn.fromNodeId)!.node.RemoveNext(oldVisualNode.node);
        }
        if (conn.fromNodeId === editingNodeId) {
          outgoingConnections.push(conn.toNodeId);
          oldVisualNode.node.RemoveNext(nodesRef.current.get(conn.toNodeId)!.node);
        }
      });

      const updatedVisualNode: VisualNode = {
        ...oldVisualNode,
        node: newNode,
      };

      setNodes((prev) => {
        const newNodes = new Map(prev);
        newNodes.set(newNode.id, updatedVisualNode);
        newNodes.delete(editingNodeId);
        nodesRef.current.set(newNode.id, updatedVisualNode);
        return newNodes;
      });

      setGenomeNode((prev) => {
        const newGenomeNode = new Map(prev);
        newGenomeNode.set(
          selectedGenomeId!,
          [...prev.get(selectedGenomeId!)!.filter((n) => n.node.id !== editingNodeId), updatedVisualNode]
        );
        return newGenomeNode;
      });

      // Restore connections
      setConnections((prev) => {
        const newConns = new Map<string, Connection>();

        incomingConnections.forEach((fromId) => {
          const fromNode = nodesRef.current.get(fromId);
          if (fromNode && newNode.CheckCompability(fromNode.node)) {
            fromNode.node.AddNext(newNode);
            const connId = uuidv4();
            newConns.set(connId, { id: connId, fromNodeId: fromId, toNodeId: newNode.id });
          }
        });

        outgoingConnections.forEach((toId) => {
          const toNode = nodesRef.current.get(toId);
          if (toNode && toNode.node.CheckCompability(newNode)) {
            newNode.AddNext(toNode.node);
            const connId = uuidv4();
            newConns.set(connId, { id: connId, fromNodeId: newNode.id, toNodeId: toId });
          }
        });

        prev.forEach((conn, connId) => {
          if (conn.fromNodeId !== editingNodeId && conn.toNodeId !== editingNodeId) {
            newConns.set(connId, conn);
          }
        });

        return newConns;
      });

      if (selectedNodeId === editingNodeId && onNodeSelect) {
        onNodeSelect(updatedVisualNode);
      }
    } else {
      // Create new node
      const newGenome = createNewGenomeWithNode(newNode);
      const pos = { x: 100 + nodes.size * 20, y: 100 + nodes.size * 20 };

      const visualNode: VisualNode = {
        node: newNode,
        position: pos,
        genomeId: newGenome.id,
        highlighted: false,
      };

      setNodes((prev) => new Map(prev).set(newNode.id, visualNode));
      setGenomes((prev) => new Map(prev).set(newGenome.id, newGenome));
      setGenomeNode((prev) => new Map(prev).set(newGenome.id, [visualNode]));
    }

    setConfigPanelOpen(false);
    setConfigNodeType(null);
    setEditingNodeId(null);
  }, [editingNodeId, selectedNodeId, selectedGenomeId, onNodeSelect, nodes, setNodes, setGenomes, setGenomeNode, setConnections]);

  const handleConfigCancel = useCallback(() => {
    setConfigPanelOpen(false);
    setConfigNodeType(null);
    setEditingNodeId(null);
  }, []);

  const handleNodeDragStart = useCallback(
    (nodeId: string, e: React.MouseEvent) => {
      e.preventDefault();
      const node = nodesRef.current.get(nodeId);
      if (!node || !svgRef.current) return;

      setNodeContextMenu(null);
      setConnectionContextMenu(null);

      const rect = svgRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const worldX = (mouseX - translate.x) / scale;
      const worldY = (mouseY - translate.y) / scale;

      setDragOffset({
        x: worldX - node.position.x,
        y: worldY - node.position.y,
      });
      setDraggingNodeId(nodeId);
    },
    [scale, translate, setDragOffset, setDraggingNodeId]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (panningRef.current) {
        // Вычисляем дельту движения курсора
        const dx = e.clientX - lastMousePosRef.current.x;
        const dy = e.clientY - lastMousePosRef.current.y;
        
        // Обновляем позицию холста
        setTranslate((prev) => ({
          x: prev.x + dx,
          y: prev.y + dy,
        }));
        
        // Сохраняем текущую позицию курсора для следующего кадра
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
        
        // Закрываем контекстные меню
        setNodeContextMenu(null);
        setConnectionContextMenu(null);
      } else if (draggingNodeId && svgRef.current) {
        // Перетаскивание ноды
        const rect = svgRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        setNodes((prev) => {
          const newNodes = new Map(prev);
          const node = newNodes.get(draggingNodeId);
          if (node) {
            node.position = {
              x: Math.round(worldX - dragOffset.x),
              y: Math.round(worldY - dragOffset.y),
            };
            newNodes.set(draggingNodeId, { ...node });
          }
          return newNodes;
        });
      }
    },
    [draggingNodeId, dragOffset, scale, translate, setTranslate, setNodes]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 2) {
        // Правая кнопка мыши - завершаем панорамирование
        panningRef.current = false;
        setIsPanning(false);
      }
      setDraggingNodeId(null);
    },
    [setIsPanning, setDraggingNodeId]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 2) {
        // Правая кнопка мыши - начинаем панорамирование
        e.preventDefault();
        panningRef.current = true;
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
        setIsPanning(true);
      }
    },
    [setIsPanning]
  );

  const handleNodeSelect = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);
      setSelectedConnectionId(null);
      const node = nodesRef.current.get(nodeId);
      if (node && onNodeSelect) {
        setSelectedGenomeId(node.genomeId);
        onNodeSelect(node);
      }
    },
    [onNodeSelect, setSelectedNodeId, setSelectedConnectionId, setSelectedGenomeId]
  );

  const handleConnectionSelect = (connectionId: string) => {
    setSelectedConnectionId(connectionId);
    setSelectedNodeId(null);
    setSelectedGenomeId(null);
  };

  const handleConnect = useCallback(
    (nodeId: string) => {
      if (connectingFrom === null) {
        setConnectingFrom(nodeId);
      } else if (connectingFrom !== nodeId) {
        const fromNode = nodesRef.current.get(connectingFrom);
        const toNode = nodesRef.current.get(nodeId);

        if (fromNode && toNode) {
          if (toNode.node.CheckCompability(fromNode.node)) {
            fromNode.node.AddNext(toNode.node);

            const connectionId = uuidv4();
            const connection: Connection = {
              id: connectionId,
              fromNodeId: connectingFrom,
              toNodeId: nodeId,
            };

            setConnections((prev) => new Map(prev).set(connectionId, connection));

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

            if (fromGenomeId !== toGenomeId) {
              const toGenomeNodes = genomeNode.get(toGenomeId)!;
              toGenomeNodes.forEach(n => n.genomeId = fromGenomeId);

              fromGenomeNode.push(...toGenomeNodes);

              genomeNode.delete(toGenomeId);
              genomes.delete(toGenomeId);
            }

            genomes.set(fromGenomeId, fromGenome);

            let isValidFlag = true;
            const inputNodes: BaseNode[] = [];
            const outputNodes: BaseNode[] = [];
            for (let node of fromGenomeNode) {
              if (node.node.previous.length === 0) {
                inputNodes.push(node.node);
                if (!(node.node instanceof InputNode)) {
                  isValidFlag = false;
                }
              }
              if (node.node.next.length === 0) {
                outputNodes.push(node.node);
                if (!(node.node instanceof OutputNode)) {
                  isValidFlag = false;
                }
              }
            }

            fromGenome.genome = new Genome(inputNodes, outputNodes);
            fromGenome.isValid = isValidFlag;

            setGenomes(new Map(genomes));
            setGenomeNode(new Map(genomeNode));

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
    },
    [connectingFrom, nodes, genomeNode, genomes, selectedNodeId, onNodeSelect, setConnectingFrom, setConnections, setNodes, setGenomeNode, setGenomes]
  );

  // Функция: Расчет позиций для нового загруженного графа с центрированием
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

  const handleLoadGenome = () => {
    loadGenomeFromFile((data: ReturnType<typeof import('../../saver/loadGenome').loadGenome>) => {
      const { nodes: loadedNodes, genome, isValid } = data;
      const loadedGenomeId = uuidv4();
      const newNodes = new Map(nodes);
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

      setNodes(newNodes);
      setGenomes((prev) => new Map(prev).set(loadedGenomeId, { id: loadedGenomeId, genome, isValid }));
      setGenomeNode((prev) => new Map(prev).set(loadedGenomeId, visualNodes));

      // Add connections from loaded genome
      loadedNodes.forEach((node: BaseNode) => {
        node.next.forEach((nextNode: BaseNode) => {
          const connId = uuidv4();
          setConnections((prev) =>
            new Map(prev).set(connId, { id: connId, fromNodeId: node.id, toNodeId: nextNode.id })
          );
        });
      });
    });
  };

  const handleGetSubgenome = () => {
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
  };

  const handleNodeContextMenu = useCallback((nodeId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setConnectionContextMenu(null);
    setNodeContextMenu({ x: e.clientX, y: e.clientY, nodeId });
  }, []);

  const handleConnectionContextMenu = useCallback((connectionId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setNodeContextMenu(null);
    setConnectionContextMenu({ x: e.clientX, y: e.clientY, connectionId });
  }, []);

  const handleNodeEdit = () => {
    if (!nodeContextMenu) return;
    const node = nodesRef.current.get(nodeContextMenu.nodeId);
    if (!node) return;

    setSelectedNodeId(nodeContextMenu.nodeId);
    openConfigPanel(node.node.GetNodeType(), nodeContextMenu.nodeId);
    setNodeContextMenu(null);
  };

  const handleNodeCopy = () => {
    if (!nodeContextMenu) return;
    const node = nodesRef.current.get(nodeContextMenu.nodeId);
    if (!node) return;

    const newNodes = new Map(nodes);
    const newGenomes = new Map(genomes);
    const newGenomeNode = new Map(genomeNode);

    const copyNode = node.node.Clone();
    const newGenomeId = uuidv4();
    const newVisualNode: VisualNode = {
      node: copyNode,
      genomeId: newGenomeId,
      position: {
        x: node.position.x + 50,
        y: node.position.y + 50,
      },
      highlighted: false,
    };

    newNodes.set(copyNode.id, newVisualNode);
    newGenomes.set(newGenomeId, {
      id: newGenomeId,
      isValid: false,
      genome: new Genome([copyNode], [copyNode]),
    });
    newGenomeNode.set(newGenomeId, [newVisualNode]);

    setNodes(newNodes);
    setGenomes(newGenomes);
    setGenomeNode(newGenomeNode);
    setNodeContextMenu(null);
  };

  const handleNodeDelete = () => {
    if (!nodeContextMenu) return;

    const nodeIdToDelete = nodeContextMenu.nodeId;
    const nodeToDelete = nodesRef.current.get(nodeIdToDelete);
    if (!nodeToDelete) return;

    const deletingNode = nodeToDelete;
    setNodeContextMenu(null);

    // Remove connections
    setConnections((prev) => {
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

        const newGenomeId = uuidv4();
        const connectedVisualNodes: VisualNode[] = [];
        const connectedNodesMap: Map<BaseNode, boolean> = new Map();
        const nodesToCheck: BaseNode[] = [node];
        const inputNodes: BaseNode[] = [];
        const outputNodes: BaseNode[] = [];
        let isValidFlag = true;

        while (nodesToCheck.length > 0) {
          const currentNode = nodesToCheck.shift()!;

          if (!connectedNodesMap.get(currentNode)) {
            if (currentNode.previous.length === 0) {
              inputNodes.push(currentNode);
              if (!(currentNode instanceof InputNode)) {
                isValidFlag = false;
              }
            }
            if (currentNode.next.length === 0) {
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

        if (connectedVisualNodes.length === 0) continue;

        const newGenome = new Genome(inputNodes, outputNodes);
        newGenomeNode.set(newGenomeId, connectedVisualNodes);
        newGenomes.set(newGenomeId, { id: newGenomeId, isValid: isValidFlag, genome: newGenome });
      }

      newGenomes.delete(nodeGenomeId);
      newGenomeNode.delete(nodeGenomeId);
      setGenomeNode(newGenomeNode);
      setGenomes(newGenomes);
    } else {
      setGenomeNode((prev) => {
        const newGenomeNode = new Map(prev);
        newGenomeNode.delete(nodeGenomeId);
        return newGenomeNode;
      });
      setGenomes((prev) => {
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
  };

  const handleConnectionDelete = () => {
    if (!connectionContextMenu) return;

    const connectionId = connectionContextMenu.connectionId;
    const connectionToDelete = connectionsRef.current.get(connectionId);
    if (!connectionToDelete) return;

    setConnectionContextMenu(null);

    setConnections((prev) => {
      const newConnections = new Map(prev);
      newConnections.delete(connectionId);
      return newConnections;
    });

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

    const toGenomeId = uuidv4();
    let toGenomeValidFlag = true;
    const toInputNodes: BaseNode[] = [];
    const toOutputNodes: BaseNode[] = [];

    let sameGenomeFlag = false;

    while (toNodesToCheck.length > 0) {
      const currentNode = toNodesToCheck.shift()!;
      if (!toConnectedNodesMap.get(currentNode)) {
        if (currentNode.id === fromNode.node.id && currentNode.id === toNode.node.id) {
          sameGenomeFlag = true;
          break;
        }
        if (currentNode.previous.length === 0) {
          toInputNodes.push(currentNode);
          if (!(currentNode instanceof InputNode)) {
            toGenomeValidFlag = false;
          }
        }
        if (currentNode.next.length === 0) {
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

      const fromGenomeId = uuidv4();
      let fromGenomeValidFlag = true;
      const fromInputNodes: BaseNode[] = [];
      const fromOutputNodes: BaseNode[] = [];

      while (fromNodesToCheck.length > 0) {
        const currentNode = fromNodesToCheck.shift()!;
        if (!fromConnectedNodesMap.get(currentNode)) {
          if (currentNode.previous.length === 0) {
            fromInputNodes.push(currentNode);
            if (!(currentNode instanceof InputNode)) {
              fromGenomeValidFlag = false;
            }
          }
          if (currentNode.next.length === 0) {
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
  };

  return (
    <div style={containerStyle}>
      <NodeToolbar
        onAddNode={(type: string) => openConfigPanel(type)}
        onLoadGenome={handleLoadGenome}
        onGetSubgenome={handleGetSubgenome}
      />

      {configPanelOpen && configNodeType && (
        <NodeConfigForm
          nodeType={configNodeType}
          existingNode={editingNodeId ? nodes.get(editingNodeId)?.node : undefined}
          onSave={handleConfigSave}
          onCancel={handleConfigCancel}
        />
      )}

      {nodeContextMenu && (
        <ContextMenu
          x={nodeContextMenu.x}
          y={nodeContextMenu.y}
          type="node"
          onEdit={handleNodeEdit}
          onCopy={handleNodeCopy}
          onDelete={handleNodeDelete}
        />
      )}

      {connectionContextMenu && (
        <ContextMenu
          x={connectionContextMenu.x}
          y={connectionContextMenu.y}
          type="connection"
          onDelete={handleConnectionDelete}
        />
      )}

      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{
          background: theme.colors.background.canvas,
          cursor: isPanning ? 'grabbing' : 'default',
        }}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseDown={handleMouseDown}
        onWheel={(e) => handleWheel(e, svgRef.current)}
        onContextMenu={(e) => e.preventDefault()}
        onClick={(e) => {
          if (e.target === svgRef.current) {
            setSelectedNodeId(null);
            setSelectedGenomeId(null);
            setSelectedConnectionId(null);
            setConnectingFrom(null);
            setNodeContextMenu(null);
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
          {Array.from(connections.values()).map((conn) => (
            <ConnectionLine
              key={conn.id}
              connection={conn}
              nodes={nodes}
              isSelected={selectedConnectionId === conn.id}
              onSelect={handleConnectionSelect}
              onContextMenu={handleConnectionContextMenu}
            />
          ))}

          {Array.from(nodes.values()).map((node) => (
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
              <NodeCard
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

const containerStyle: CSSProperties = {
  width: '100%',
  height: '100%',
  position: 'relative',
  backgroundColor: theme.colors.background.canvas,
};
