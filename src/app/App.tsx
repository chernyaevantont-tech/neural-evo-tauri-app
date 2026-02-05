import React, { useState } from 'react';
import { VisualNode, VisualGenome } from '../shared/types';
import { NetworkCanvas, SidePanel } from '../widgets';
import { TitleBar } from '../widgets/title-bar';
import { theme } from '../shared/lib';

function App() {
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
  const genomesState = useState<Map<string, VisualGenome>>(new Map());

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        backgroundColor: theme.colors.background.primary,
        fontFamily: theme.typography.fontFamily,
        color: theme.colors.text.primary,
        margin: 0,
        padding: 0,
      }}
    >
      <TitleBar />
      
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        minHeight: 0,
      }}>
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
