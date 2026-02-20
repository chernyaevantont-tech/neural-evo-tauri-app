import React, { CSSProperties, RefObject, useEffect, useRef } from 'react';
import { Node, useCanvasGenomeStore } from '../../entities/canvas-genome';
import { ConnectionLine } from '../../entities/canvas-genome';
import { theme } from '../../shared/lib';
import { MenuType } from '../side-menu/SideMenu';
import { useCanvasStateStore } from '../../entities/canvas-state';
import { ContextMenu } from '../../shared/ui/ContextMenu/ContextMenu';
import { CopyNodeContextMenuItem } from '../../features/copy-node';
import { EditNodeContextMenuItem } from '../../features/edit-node';
import { DeleteNodeContextMenuItem } from '../../features/delete-node';
import { DeleteGenomeContextMenuItem } from '../../features/delete-genome';
import { DeleteConnectionContextMenuItem } from '../../features/delete-connection';
import { useContinueMovingNode, useEndDraggingNode, useStartDraggingNode } from '../../features/dragging-move-node';
import { useContinueCanvasPanning, useEndCanvasPanning, useStartCanvasPanning } from '../../features/canvas-panning';
import { useResizeCanvas } from '../../features/resize-canvas';
import { useConnectNodes } from '../../features/connect-nodes';
import { useCanvasSelectedBreed } from '../../features/breed-genomes';
import { ConnectionContextMenu } from './ConnectionContextMenu/ConnectionContextMenu';
import { NodeContextMenu } from './NodeContextMenu/NodeContextMenu';
import { GenomeContextMenu } from './GenomContextMenu/GenomeContextMenu';
import { useOnClickOutside } from './hooks';

interface NetworkCanvasProps {
  menuType: MenuType;
}

export const NetworkCanvas: React.FC<NetworkCanvasProps> = ({
  menuType
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const isPanning = useCanvasStateStore(state => state.isPanning);
  const selectedNodeId = useCanvasStateStore(state => state.selectedNodeId);
  const selectedConnectionId = useCanvasStateStore(state => state.selectedConnectionId);
  const translate = useCanvasStateStore(state => state.translate);
  const scale = useCanvasStateStore(state => state.scale);
  const setCanvasWidth = useCanvasStateStore(state => state.setCanvasWidth);
  const setCanvasHeight = useCanvasStateStore(state => state.setCanvasHeight);
  const setNodeContextMenu = useCanvasStateStore(state => state.setNodeContextMenu);
  const setConnectionContextMenu = useCanvasStateStore(state => state.setConnectionContextMenu);
  const setGenomeContextMenu = useCanvasStateStore(state => state.setGenomeContextMenu);
  const setSelectedNodeId = useCanvasStateStore(state => state.setSelectedNodeId);
  const setSelectedConnectionId = useCanvasStateStore(state => state.setSelectedConnectionId);
  const setSelectedGenomeId = useCanvasStateStore(state => state.setSelectedGenomeId);

  const nodes = useCanvasGenomeStore(state => state.nodes);
  const connections = useCanvasGenomeStore(state => state.connections);

  const canvasStartMovingNodeHandle = useStartDraggingNode();
  const canvasMovingNodeHandle = useContinueMovingNode();
  const canvasEndMovingNodeHandle = useEndDraggingNode();

  const canvasStartPaddingHandle = useStartCanvasPanning();
  const canvasPaddingHandle = useContinueCanvasPanning();
  const canvasEndPaddingHandle = useEndCanvasPanning();

  const resizeCanvas = useResizeCanvas();

  const connectNodes = useConnectNodes();
  const breedGenomes = useCanvasSelectedBreed();

  useEffect(() => {
    if (!svgRef || !svgRef.current) return;
    setCanvasWidth(svgRef.current.width.baseVal.value)
    setCanvasHeight(svgRef.current.height.baseVal.value)
  }, [svgRef.current?.width, svgRef.current?.height]);

  useOnClickOutside(svgRef, () => {
    // setNodeContextMenu(null);
  });

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button == 2) {
      canvasStartPaddingHandle(e.clientX, e.clientY);
    }
    setNodeContextMenu(null);
  }

  const onMouseMove = (e: React.MouseEvent) => {
    if (!svgRef.current) return;
    canvasMovingNodeHandle(
      svgRef.current.clientLeft,
      svgRef.current.clientTop,
      e.clientX,
      e.clientY
    );
    canvasPaddingHandle(
      e.clientX,
      e.clientY
    );
  }

  const onMouseUp = (e: React.MouseEvent) => {
    canvasEndMovingNodeHandle();
    canvasEndPaddingHandle();
  }

  const onWheel = (e: React.WheelEvent) => {
    setNodeContextMenu(null);
    if (!svgRef.current) return;
    resizeCanvas(
      e.clientX,
      e.clientY,
      e.deltaY,
      svgRef.current.clientLeft,
      svgRef.current.clientTop,
    )
  }

  const onDragStart = (nodeId: string, e: React.MouseEvent) => {
    if (!svgRef.current) return;
    canvasStartMovingNodeHandle(
      nodeId,
      svgRef.current.clientLeft,
      svgRef.current.clientTop,
      e.clientX,
      e.clientY
    )
  }

  return (
    <div style={containerStyle}>

      <NodeContextMenu />
      <ConnectionContextMenu />
      <GenomeContextMenu />

      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{
          background: theme.colors.background.canvas,
          cursor: isPanning ? 'grabbing' : 'default',
        }}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseDown={onMouseDown}
        onWheel={onWheel}
        onContextMenu={(e) => e.preventDefault()}
        onClick={(e) => {
          if (e.target === svgRef.current) {
            setSelectedNodeId(null);
            connectNodes(null);
            setNodeContextMenu(null);
            setConnectionContextMenu(null);
            setGenomeContextMenu(null);
            breedGenomes(null);
          } else {
            setNodeContextMenu(null);
          }
        }}
        onBlur={() => {
          setNodeContextMenu(null);
        }}
      >
        <g transform={`translate(${translate.x}, ${translate.y}) scale(${scale})`}>
          {Array.from(connections.values()).map((conn) => (
            <ConnectionLine
              key={conn.id}
              connection={conn}
              nodes={nodes}
              isSelected={selectedConnectionId === conn.id}
              onSelect={(id: string) => {
                if (menuType == "Layers") {
                  setSelectedConnectionId(id);
                } else {
                  const genomeId = nodes.get(conn.fromNodeId)?.genomeId;
                  if (!genomeId) return;
                  setSelectedGenomeId(genomeId);
                }
              }}
              onContextMenu={(id: string, e: React.MouseEvent) => {
                e.preventDefault();
                if (menuType == "Layers") {
                  setConnectionContextMenu({ x: e.clientX, y: e.clientY, connectionId: id })
                } else {
                  const genomeId = nodes.get(conn.fromNodeId)?.genomeId;
                  if (!genomeId) return;
                  setGenomeContextMenu({ x: e.clientX, y: e.clientY, genomeId: genomeId })
                }
              }}
            />
          ))}

          {Array.from(nodes.values()).map((node) => (
            <g
              key={node.node.id}
              onClickCapture={(e) => {
                if (e.shiftKey) {
                  e.stopPropagation();
                  if (menuType == "Layers") {
                    connectNodes(node.node.id);
                  } else {
                    breedGenomes(node.genomeId);
                  }
                }
              }}
            >
              <Node
                node={node}
                isSelected={selectedNodeId === node.node.id}
                onSelect={(id: string) => {
                  if (menuType == "Layers") {
                    setSelectedNodeId(id);
                  } else {
                    const genomeId = node.genomeId;
                    setSelectedGenomeId(genomeId);
                  }

                }
                }
                onDragStart={onDragStart}
                onContextMenu={(id: string, e: React.MouseEvent) => {
                  e.preventDefault();
                  if (menuType == "Layers") {
                    setNodeContextMenu({ x: e.clientX, y: e.clientY, nodeId: id })
                  } else {
                    const genomeId = node.genomeId;
                    setGenomeContextMenu({ x: e.clientX, y: e.clientY, genomeId: genomeId })
                  }
                }}
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
