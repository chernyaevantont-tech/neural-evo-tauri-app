import React from 'react';
import { Connection, VisualNode } from '../../../shared/types';
import { theme } from '../../../shared/lib';
import styles from './ConnectionLine.module.css';

interface ConnectionLineProps {
  connection: Connection;
  nodes: Map<string, VisualNode>;
  isSelected: boolean;
  onSelect: (connectionId: string) => void;
  onContextMenu: (id: string, e: React.MouseEvent) => void;
}

export const ConnectionLine: React.FC<ConnectionLineProps> = ({
  connection,
  nodes,
  isSelected,
  onSelect,
  onContextMenu,
}) => {
  const fromNode = nodes.get(connection.fromNodeId);
  const toNode = nodes.get(connection.toNodeId);

  if (!fromNode || !toNode) return null;

  const radius = 50;
  const startX = fromNode.position.x + radius;
  const startY = fromNode.position.y + radius;
  const endX = toNode.position.x + radius;
  const endY = toNode.position.y + radius;

  const dx = endX - startX;
  const dy = endY - startY;
  const angle = Math.atan2(dy, dx);
  const distance = Math.sqrt(dx * dx + dy * dy);

  const adjustedStartX = startX + radius * Math.cos(angle);
  const adjustedStartY = startY + radius * Math.sin(angle);
  const adjustedEndX = endX - radius * Math.cos(angle);
  const adjustedEndY = endY - radius * Math.sin(angle);

  if (distance < radius * 2) return null;

  const arrowSize = 15;
  const arrowTip = { x: adjustedEndX, y: adjustedEndY };
  const arrowBase1 = {
    x: adjustedEndX - arrowSize * Math.cos(angle - Math.PI / 6),
    y: adjustedEndY - arrowSize * Math.sin(angle - Math.PI / 6),
  };
  const arrowBase2 = {
    x: adjustedEndX - arrowSize * Math.cos(angle + Math.PI / 6),
    y: adjustedEndY - arrowSize * Math.sin(angle + Math.PI / 6),
  };

  const strokeColor = isSelected 
    ? theme.colors.border.focus 
    : theme.colors.text.secondary;

  return (
    <div className={styles.connectionLine}>
      <line
        x1={adjustedStartX}
        y1={adjustedStartY}
        x2={adjustedEndX}
        y2={adjustedEndY}
        stroke={strokeColor}
        strokeWidth={isSelected ? 3 : 2}
        opacity={0.8}
      />
      <polygon
        points={`${arrowTip.x},${arrowTip.y} ${arrowBase1.x},${arrowBase1.y} ${arrowBase2.x},${arrowBase2.y}`}
        fill={strokeColor}
        opacity={0.8}
      />
    </div>
  );
};
