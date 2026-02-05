import { useState } from "react";
import "./App.css";
import { TitleBar } from "./widgets/title-bar";
import { NetworkCanvas } from "./widgets";
import { SidePanel } from "./widgets";
import { theme } from "./shared/lib";
import type { VisualNode, VisualGenome } from "./shared/types";

function App() {
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
  const [genomes, setGenomes] = useState<Map<string, VisualGenome>>(new Map());

  console.log('=== APP RENDERING ===');
  console.log('App component is rendering with TitleBar');

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      width: '100vw',
      height: '100vh',
      overflow: 'hidden',
      backgroundColor: theme.colors.background.primary,
      margin: 0,
      padding: 0,
    }}>
      {/* DEBUG: Simple title bar */}
      <div style={{
        height: '32px',
        minHeight: '32px',
        backgroundColor: '#ff0000',
        color: '#ffffff',
        display: 'flex',
        alignItems: 'center',
        paddingLeft: '12px',
        flexShrink: 0,
      }}>
        DEBUG: Title Bar Here
      </div>
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        minHeight: 0,
      }}>
        <div style={{
          flex: 1,
          position: 'relative',
          height: '100%'
        }}>
          <NetworkCanvas 
            onNodeSelect={setSelectedNode} 
            onGenomeSelect={setSelectedGenome} 
            genomesState={[genomes, setGenomes]}
          />
        </div>
        
        <SidePanel 
          selectedNode={selectedNode} 
          genomes={Array.from(genomes.values())} 
        />
      </div>
    </div>
  );
}

export default App;
