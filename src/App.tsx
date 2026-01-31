import { useState } from "react";
import "./App.css";
import { NetworkEditor } from "./components/NetworkEditor";
import { NodeInfoPanel } from "./components/NodeInfoPanel";
import { VisualGenome, VisualNode } from "./components/types";
import { GenomeInfoPanel } from "./components/GenomeInfoPanel";

function App() {
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
  const [genomes, setGenomes] = useState<Map<string, VisualGenome>>(new Map());

  return (
    <div style={{
      display: 'flex',
      width: '100vw',
      height: '100vh',
      overflow: 'hidden'
    }}>
      <div style={{
        flex: 1,
        position: 'relative',
        height: '100%'
      }}>
        <NetworkEditor onNodeSelect={setSelectedNode} onGenomeSelect={setSelectedGenome} genomes={genomes} setGenomes={setGenomes}/>
      </div>
      
      <div style={{
        width: '400px',
        height: '100%',
        padding: '20px',
        background: '#fafafa',
        borderLeft: '1px solid #ddd',
        overflow: 'auto',
        display: "flex",
        flexDirection: "column",
        gap: "20px",
      }}>
        <NodeInfoPanel selectedNode={selectedNode} />
        <GenomeInfoPanel genomes={Array.from(genomes.values())}/>
      </div>
    </div>
  );
}

export default App;
