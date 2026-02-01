import React from 'react';
import { Connection, VisualNode } from './types';

interface ConnectionRendererProps {
    connection: Connection;
    nodes: Map<string, VisualNode>;
    isSelected: boolean;
    onSelect: (connectionId: string) => void;
    onContextMenu: (id: string, e: React.MouseEvent) => void;
}

export const ConnectionRenderer: React.FC<ConnectionRendererProps> = ({
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

    // Calculate the angle from start to end
    const dx = endX - startX;
    const dy = endY - startY;
    const angle = Math.atan2(dy, dx);
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Adjust start and end points to be on the circle edge
    const adjustedStartX = startX + radius * Math.cos(angle);
    const adjustedStartY = startY + radius * Math.sin(angle);
    const adjustedEndX = endX - radius * Math.cos(angle);
    const adjustedEndY = endY - radius * Math.sin(angle);

    // Only draw if there's actual distance
    if (distance < radius * 2) return null;

    // Arrow size
    const arrowSize = 15;
    
    // Calculate arrow points
    const arrowTip = { x: adjustedEndX, y: adjustedEndY };
    const arrowBase1 = {
        x: adjustedEndX - arrowSize * Math.cos(angle - Math.PI / 6),
        y: adjustedEndY - arrowSize * Math.sin(angle - Math.PI / 6)
    };
    const arrowBase2 = {
        x: adjustedEndX - arrowSize * Math.cos(angle + Math.PI / 6),
        y: adjustedEndY - arrowSize * Math.sin(angle + Math.PI / 6)
    };

    return (
        <g onMouseDown={(e) => {
            e.stopPropagation();
            onSelect(connection.id);
        }}
            onContextMenu={(e) => {
                e.stopPropagation();
                onContextMenu(connection.id, e);
            }}
        >
            <line
                x1={adjustedStartX}
                y1={adjustedStartY}
                x2={adjustedEndX}
                y2={adjustedEndY}
                stroke={isSelected ? '#FFD700' : '#333'}
                strokeWidth={3}
            />
            <polygon
                points={`${arrowTip.x},${arrowTip.y} ${arrowBase1.x},${arrowBase1.y} ${arrowBase2.x},${arrowBase2.y}`}
                fill={isSelected ? '#FFD700' : '#333'}
            />
        </g>
    );
};

