import React from 'react';
import { NodeInfoCard, useCanvasGenomeStore } from '../../entities/canvas-genome';
import styles from './SidePanel.module.css';
import { MenuType } from '../side-menu/SideMenu';
import { GenomeCard } from '../../entities/canvas-genome/ui/GenomeCard/GenomeCard';
import { useCanvasStateStore } from '../../entities/canvas-state';
import { SaveGenomeButton } from '../../features/genome-save-load/ui/SaveGenomeButton';

interface SidePanelProps {
  menuType: MenuType;
}

import { EvolutionManager } from '../../features/evolution-manager';

export const SidePanel: React.FC<SidePanelProps> = ({ menuType }) => {
  const genomes = Array.from(useCanvasGenomeStore(state => state.genomes).values());
  const nodes = useCanvasGenomeStore(state => state.nodes);

  const selectedNodeId = useCanvasStateStore(state => state.selectedNodeId);

  const selectedNode = selectedNodeId ? nodes.get(selectedNodeId) : null;

  return (
    <div className={styles.container}>
      {
        menuType == "Layers" && selectedNode && <NodeInfoCard node={selectedNode} />
      }
      {/* <div className={styles.divider} /> */}
      {
        menuType == "Genomes" && (
          <>
            <div className={styles.genomeListContainer}>
              <h3 className={styles.title}>Genomes</h3>
              <div className={styles.list}>
                {genomes.map((genome) => (
                  <GenomeCard key={genome.id} genomeId={genome.id} isValid={genome.isValid} actionSlot={<SaveGenomeButton genome={genome.genome} />} />
                ))}
              </div>
            </div>
            <EvolutionManager />
          </>
        )
      }
    </div>
  );
};
