import React, { useState } from 'react';
import { VisualNode, VisualGenome } from '../shared/types';
import { NetworkCanvas, SidePanel } from '../widgets';
import { TitleBar } from '../widgets';
import { SideMenu } from '../widgets/side-menu/SideMenu';
import styles from './App.module.css';

function App() {
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
  const genomesState = useState<Map<string, VisualGenome>>(new Map());

  return (
    <div className={styles.container}>
      <TitleBar />
      
      <div className={styles.content}>
        <SideMenu/>

        <NetworkCanvas
          onNodeSelect={setSelectedNode}
          onGenomeSelect={setSelectedGenome}
          genomesState={genomesState}
        />
        
        <SidePanel
          selectedNode={selectedNode}
          genomes={Array.from(genomesState[0].values())}
        />
      </div>
    </div>
  );
}

export default App;
