import { Dispatch, SetStateAction, useState } from "react";
import { Connection, VisualGenome, VisualNode } from "../../components/types";

export type NetworkStateType = {
    nodes: Map<string, VisualNode>,
    setNodes: Dispatch<SetStateAction<Map<string, VisualNode>>>;
    genomeNode: Map<string, VisualNode[]>;
    setGenomeNode: Dispatch<SetStateAction<Map<string, VisualNode[]>>>;
    connections: Map<string, Connection>;
    setConnections: Dispatch<SetStateAction<Map<string, Connection>>>;
    genomes: Map<string, VisualGenome>;
    setGenomes: Dispatch<SetStateAction<Map<string, VisualGenome>>>;
    selectedNodeId: string | null;
    setSelectedNodeId: Dispatch<SetStateAction<string|null>>;
    selectedGenomeId: string | null;
    setSelectedGenomeId: Dispatch<SetStateAction<string | null>>;
    selectedConnectionId: string | null;
    setSelectedConnectionId: Dispatch<SetStateAction<string | null>>;
};

export const useNetworkState = () : NetworkStateType => {
  const [nodes, setNodes] = useState<Map<string, VisualNode>>(new Map());
  const [genomeNode, setGenomeNode] = useState<Map<string, VisualNode[]>>(new Map());
  const [connections, setConnections] = useState<Map<string, Connection>>(new Map());
  const [genomes, setGenomes] = useState<Map<string, VisualGenome>>(new Map());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedGenomeId, setSelectedGenomeId] = useState<string | null>(null);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);

  return {
    nodes,
    setNodes,
    genomeNode,
    setGenomeNode,
    connections,
    setConnections,
    genomes,
    setGenomes,
    selectedNodeId,
    setSelectedNodeId,
    selectedGenomeId,
    setSelectedGenomeId,
    selectedConnectionId,
    setSelectedConnectionId,
  };
};