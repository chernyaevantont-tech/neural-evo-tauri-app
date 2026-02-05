import React, { CSSProperties } from 'react';
import { Button, LoadIcon, PlusIcon } from '../../shared/ui';
import { theme } from '../../shared/lib';

interface NodeToolbarProps {
  onAddNode: (nodeType: string) => void;
  onLoadGenome: () => void;
  onGetSubgenome: () => void;
}

export const NodeToolbar: React.FC<NodeToolbarProps> = ({
  onAddNode,
  onLoadGenome,
  onGetSubgenome,
}) => {
  const nodeTypes = [
    { type: 'Input', label: 'Input' },
    { type: 'Dense', label: 'Dense' },
    { type: 'Conv2D', label: 'Conv2D' },
    { type: 'Pooling', label: 'Pooling' },
    { type: 'Flatten', label: 'Flatten' },
    { type: 'Add', label: 'Add' },
    { type: 'Concat2D', label: 'Concat' },
    { type: 'Output', label: 'Output' },
  ];

  return (
    <div style={containerStyle}>
      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Add Layers</h4>
        <div style={buttonGridStyle}>
          {nodeTypes.map(({ type, label }) => (
            <Button
              key={type}
              onClick={() => onAddNode(type)}
              variant="secondary"
              size="sm"
              icon={<PlusIcon size={14} />}
              style={nodeButtonStyle}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      <div style={dividerStyle} />

      <div style={sectionStyle}>
        <h4 style={sectionTitleStyle}>Genome Operations</h4>
        <div style={operationsStyle}>
          <Button
            onClick={onLoadGenome}
            variant="primary"
            size="md"
            icon={<LoadIcon size={16} />}
            fullWidth
          >
            Load Genome
          </Button>
          <Button
            onClick={onGetSubgenome}
            variant="secondary"
            size="md"
            fullWidth
          >
            Get Subgenome
          </Button>
        </div>
      </div>
    </div>
  );
};

const containerStyle: CSSProperties = {
  position: 'absolute',
  top: theme.spacing.md,
  left: theme.spacing.md,
  zIndex: 10,
  backgroundColor: theme.colors.background.secondary,
  padding: theme.spacing.lg,
  borderRadius: theme.borderRadius.lg,
  boxShadow: theme.shadows.lg,
  border: `1px solid ${theme.colors.border.primary}`,
  maxWidth: '280px',
};

const sectionStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing.sm,
};

const sectionTitleStyle: CSSProperties = {
  margin: 0,
  fontSize: theme.typography.fontSize.sm,
  fontWeight: theme.typography.fontWeight.semibold,
  color: theme.colors.text.secondary,
  textTransform: 'uppercase',
  fontFamily: theme.typography.fontFamily,
  letterSpacing: '0.5px',
};

const buttonGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, 1fr)',
  gap: theme.spacing.xs,
};

const nodeButtonStyle: CSSProperties = {
  justifyContent: 'flex-start',
};

const dividerStyle: CSSProperties = {
  height: '1px',
  backgroundColor: theme.colors.border.primary,
  margin: `${theme.spacing.lg} 0`,
};

const operationsStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing.xs,
};
