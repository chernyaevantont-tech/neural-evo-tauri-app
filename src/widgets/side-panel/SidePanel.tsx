import React, { useState } from 'react';
import { VisualNode, VisualGenome } from '../../shared/types';
import { NodeInfoCard } from '../../entities/node/ui';
import { GenomeList } from '../../entities/genome/ui';
import styles from './SidePanel.module.css';
import { MenuType } from '../side-menu/SideMenu';

interface SidePanelProps {
  selectedNode: VisualNode | null;
  genomes: VisualGenome[];
  menuType: MenuType;
}

export const SidePanel: React.FC<SidePanelProps> = ({ selectedNode, genomes, menuType }) => {

  return (
    <div className={styles.container}>
      {
        menuType == "Layers" && <NodeInfoCard node={selectedNode} />
      }
      {/* <div className={styles.divider} /> */}
      {
        menuType == "Genomes" && <GenomeList genomes={genomes} />
      }
    </div>
  );
};
