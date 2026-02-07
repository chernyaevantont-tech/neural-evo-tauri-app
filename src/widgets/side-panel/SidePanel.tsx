import React, { useState } from 'react';
import { VisualNode, VisualGenome } from '../../shared/types';
import { NodeInfoCard } from '../../entities/node/ui';
import { GenomeList } from '../../entities/genome/ui';
import styles from './SidePanel.module.css';

interface SidePanelProps {
  selectedNode: VisualNode | null;
  genomes: VisualGenome[];
}

export const SidePanel: React.FC<SidePanelProps> = ({ selectedNode, genomes }) => {

  return (
    <div className={styles.container}>
      <NodeInfoCard node={selectedNode} />
      
      <div className={styles.divider} />
      
      <GenomeList genomes={genomes} />
    </div>
  );
};
