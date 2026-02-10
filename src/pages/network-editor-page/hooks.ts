import { Dispatch, SetStateAction, useCallback, useState } from "react";
import { Connection, VisualGenome, VisualNode } from "../../components/types";
import { Position } from "../../shared/types";

export type NetworkStateType = {
  nodes: Map<string, VisualNode>,
  setNodes: Dispatch<SetStateAction<Map<string, VisualNode>>>;
  genomeNode: Map<string, VisualNode[]>;
  setGenomeNode: Dispatch<SetStateAction<Map<string, VisualNode[]>>>;
  connections: Map<string, Connection>;
  setConnections: Dispatch<SetStateAction<Map<string, Connection>>>;
  genomes: Map<string, VisualGenome>;
  setGenomes: Dispatch<SetStateAction<Map<string, VisualGenome>>>;
  selectedNodeId: string | null;
  setSelectedNodeId: Dispatch<SetStateAction<string | null>>;
  selectedGenomeId: string | null;
  setSelectedGenomeId: Dispatch<SetStateAction<string | null>>;
  selectedConnectionId: string | null;
  setSelectedConnectionId: Dispatch<SetStateAction<string | null>>;
};

export const useNetworkState = (): NetworkStateType => {
  const [nodes, setNodes] = useState<Map<string, VisualNode>>(new Map());
  const [genomeNode, setGenomeNode] = useState<Map<string, VisualNode[]>>(new Map());
  const [connections, setConnections] = useState<Map<string, Connection>>(new Map());
  const [genomes, setGenomes] = useState<Map<string, VisualGenome>>(new Map());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedGenomeId, setSelectedGenomeId] = useState<string | null>(null);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);

  return {
    nodes,
    setNodes,
    genomeNode,
    setGenomeNode,
    connections,
    setConnections,
    genomes,
    setGenomes,
    selectedNodeId,
    setSelectedNodeId,
    selectedGenomeId,
    setSelectedGenomeId,
    selectedConnectionId,
    setSelectedConnectionId,
  };
};

export type CanvasInteractionType = {
  draggingNodeId: string | null;
  setDraggingNodeId: Dispatch<SetStateAction<string | null>>;
  connectingFrom: string | null;
  setConnectingFrom: Dispatch<SetStateAction<string | null>>;
  dragOffset: Position;
  setDragOffset: Dispatch<SetStateAction<Position>>;
  scale: number;
  setScale: Dispatch<SetStateAction<number>>;
  translate: Position;
  setTranslate: Dispatch<SetStateAction<Position>>;
  isPanning: boolean;
  setIsPanning: Dispatch<SetStateAction<boolean>>;
  panStart: Position;
  setPanStart: Dispatch<SetStateAction<Position>>;
  handleWheel: (e: React.WheelEvent, svgElement: SVGSVGElement | null) => void;
}

export const useCanvasInteraction = () => {
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
  const [scale, setScale] = useState<number>(1);
  const [translate, setTranslate] = useState<Position>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [panStart, setPanStart] = useState<Position>({ x: 0, y: 0 });

  const handleWheel = useCallback(
    (e: React.WheelEvent, svgElement: SVGSVGElement | null) => {
      if (!svgElement) return;

      const rect = svgElement.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const worldX = (mouseX - translate.x) / scale;
      const worldY = (mouseY - translate.y) / scale;

      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.1, Math.min(5, scale * delta));

      const newTranslateX = mouseX - worldX * newScale;
      const newTranslateY = mouseY - worldY * newScale;

      setScale(newScale);
      setTranslate({ x: newTranslateX, y: newTranslateY });
    },
    [scale, translate]
  );

  return {
    draggingNodeId,
    setDraggingNodeId,
    connectingFrom,
    setConnectingFrom,
    dragOffset,
    setDragOffset,
    scale,
    setScale,
    translate,
    setTranslate,
    isPanning,
    setIsPanning,
    panStart,
    setPanStart,
    handleWheel,
  };
};