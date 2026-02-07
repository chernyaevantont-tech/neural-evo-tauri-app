import React from 'react';
import { VisualNode } from './types';
import styles from './NodeRenderer.module.css';

interface NodeRendererProps {
    node: VisualNode;
    isSelected: boolean;
    onSelect: (id: string) => void;
    onDragStart: (id: string, e: React.MouseEvent) => void;
    onContextMenu: (id: string, e: React.MouseEvent) => void;
}

const getNodeColor = (type: String): string => {
    switch (type) {
        case 'Input': return '#4CAF50';
        case 'Dense': return '#2196F3';
        case 'Conv2D': return '#FF9800';
        case 'Pooling': return '#9C27B0';
        case "Flatten": return '#43672e'
        case 'Add': return '#F44336';
        case 'Concat2D': return '#E91E63';
        case 'Output': return '#ff0000'
        default: return '#757575';
    }
};

const getNodeLabel = (type: String): string => {
    switch (type) {
        case 'Conv2D': return 'Conv2D';
        case 'Concat2D': return 'Concat';
        default: return type.toString();
    }
};

export const NodeRenderer: React.FC<NodeRendererProps> = ({
    node,
    isSelected,
    onSelect,
    onDragStart,
    onContextMenu
}) => {
    const radius = 50;
    const color = getNodeColor(node.node.GetNodeType());
    const label = getNodeLabel(node.node.GetNodeType());

    return (
        <div className={styles.nodeRenderer}
            onMouseDown={(e) => {
                e.stopPropagation();
                onDragStart(node.node.id, e);
                onSelect(node.node.id);
            }}
            onContextMenu={(e) => {
                e.stopPropagation();
                onContextMenu(node.node.id, e);
            }}
        >
            <circle
                cx={radius}
                cy={radius}
                r={radius}
                fill={color}
                stroke={isSelected ? '#FFD700' : '#333'}
                strokeWidth={isSelected ? 4 : 2}
                opacity={0.9}
            />
            <circle
                cx={radius}
                cy={radius}
                r={radius}
                fill={node.highlighted ? 'rgb(119, 0, 255)' : 'transparent'}
                opacity={0.7}
                pointerEvents="none"
            />
            <text
                x={radius}
                y={radius}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontSize="14"
                fontWeight="bold"
                pointerEvents="none"
            >
                {label}
            </text>
        </div>
    );
};

