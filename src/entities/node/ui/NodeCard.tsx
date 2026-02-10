import React from 'react';
import { VisualNode } from '../../../shared/types';
import { getNodeColor, getNodeLabel, theme } from '../../../shared/lib';

interface NodeCardProps {
  node: VisualNode;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onDragStart: (id: string, e: React.MouseEvent) => void;
  onContextMenu: (id: string, e: React.MouseEvent) => void;
}

export const NodeCard: React.FC<NodeCardProps> = ({
  node,
  isSelected,
  onSelect,
  onDragStart,
  onContextMenu,
}) => {
  const radius = 50;
  const color = getNodeColor(node.node.GetNodeType());
  const label = getNodeLabel(node.node.GetNodeType());

  return (
    <g
      transform={`translate(${node.position.x}, ${node.position.y})`}
      onMouseDown={(e) => {
        e.stopPropagation();
        onDragStart(node.node.id, e);
        onSelect(node.node.id);
      }}
      onContextMenu={(e) => {
        e.stopPropagation();
        onContextMenu(node.node.id, e);
      }}
      style={{ cursor: 'move', userSelect: 'none' }}
    >
      {/* Main circle */}
      <circle
        cx={radius}
        cy={radius}
        r={radius}
        fill={color}
        stroke={isSelected ? theme.colors.border.focus : theme.colors.border.primary}
        strokeWidth={isSelected ? 3 : 2}
        opacity={0.95}
        filter="drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))"
      />
      
      {/* Highlight overlay */}
      {node.highlighted && (
        <circle
          cx={radius}
          cy={radius}
          r={radius}
          fill={theme.colors.accent.primary}
          opacity={0.9}
          pointerEvents="none"
        />
      )}
      
      {/* Label */}
      <text
        x={radius}
        y={radius}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="white"
        fontSize={theme.typography.fontSize.md}
        fontWeight={theme.typography.fontWeight.semibold}
        fontFamily={theme.typography.fontFamily}
        pointerEvents="none"
      >
        {label}
      </text>
    </g>
  );
};
