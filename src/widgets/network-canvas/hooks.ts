import { useCallback, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Position, VisualGenome, VisualNode } from '../../shared/types';
import { BaseNode } from '../../evo/nodes/base_node';
import { Genome } from '../../evo/genome';
import { InputNode } from '../../evo/nodes/layers/input_node';
import { OutputNode } from '../../evo/nodes/layers/output_node';

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

export const createNewGenomeWithNode = (node: BaseNode): VisualGenome => {
  return {
    id: uuidv4(),
    genome: new Genome([node], [node]),
    isValid: false,
  };
};

export const updateGenomeValidity = (
  genomeNodes: VisualNode[],
  genome: Genome
): { isValid: boolean; inputNodes: BaseNode[]; outputNodes: BaseNode[] } => {
  let isValid = true;
  const inputNodes: BaseNode[] = [];
  const outputNodes: BaseNode[] = [];

  for (let visualNode of genomeNodes) {
    const node = visualNode.node;
    if (node.previous.length === 0) {
      inputNodes.push(node);
      if (!(node instanceof InputNode)) {
        isValid = false;
      }
    }
    if (node.next.length === 0) {
      outputNodes.push(node);
      if (!(node instanceof OutputNode)) {
        isValid = false;
      }
    }
  }

  genome.inputNodes = inputNodes;
  genome.outputNodes = outputNodes;

  return { isValid, inputNodes, outputNodes };
};
