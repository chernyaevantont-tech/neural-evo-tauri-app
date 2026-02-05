import React, { CSSProperties } from 'react';
import { VisualNode, VisualGenome } from '../../shared/types';
import { NodeInfoCard } from '../../entities/node/ui';
import { GenomeList } from '../../entities/genome/ui';
import { theme } from '../../shared/lib';

interface SidePanelProps {
  selectedNode: VisualNode | null;
  genomes: VisualGenome[];
}

export const SidePanel: React.FC<SidePanelProps> = ({ selectedNode, genomes }) => {
  return (
    <div style={containerStyle}>
      <NodeInfoCard node={selectedNode} />
      
      <div style={dividerStyle} />
      
      <GenomeList genomes={genomes} />
    </div>
  );
};

const containerStyle: CSSProperties = {
  width: '380px',
  height: '100%',
  padding: theme.spacing.lg,
  backgroundColor: theme.colors.background.secondary,
  borderLeft: `1px solid ${theme.colors.border.primary}`,
  overflow: 'auto',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing.lg,
};

const dividerStyle: CSSProperties = {
  height: '1px',
  backgroundColor: theme.colors.border.primary,
  margin: `${theme.spacing.md} 0`,
};
