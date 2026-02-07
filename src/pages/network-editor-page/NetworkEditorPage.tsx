import { useCallback, useState } from 'react';
import styles from './NetworkEditorPage.module.css';
import { Connection, VisualGenome, VisualNode } from '../../components/types';
import { NetworkCanvas, SidePanel, TitleBar } from '../../widgets';
import { SideMenu } from '../../widgets/side-menu/SideMenu';
import { useNetworkState } from './hooks';
import { BaseNode } from '../../evo/nodes/base_node';
import { v4 } from 'uuid';
import { Genome } from '../../evo/genome';

export const NetworkEditorPage: React.FC = () => {
    const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
    const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);
    const genomesState = useState<Map<string, VisualGenome>>(new Map());
    const networkState = useNetworkState();
    const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
    const [configNodeType, setConfigNodeType] = useState<string | null>(null);
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);

    const {
        nodes,
        setNodes,
        genomeNode,
        setGenomeNode,
        connections,
        setConnections,
        selectedNodeId,
        setSelectedNodeId,
        selectedGenomeId,
        setSelectedGenomeId,
        selectedConnectionId,
        setSelectedConnectionId,
    } = networkState;

    const handleAddNode = useCallback((newNode: BaseNode) => {
        const newGenome = {
            id: v4(),
            genome: new Genome([newNode], [newNode]),
            isValid: false,
        };
        const pos = { x: 100 + nodes.size * 20, y: 100 + nodes.size * 20 };

        const visualNode: VisualNode = {
            node: newNode,
            position: pos,
            genomeId: newGenome.id,
            highlighted: false,
        };

        setNodes((prev) => new Map(prev).set(newNode.id, visualNode));
        genomesState[1]((prev) => new Map(prev).set(newGenome.id, newGenome));
        setGenomeNode((prev) => new Map(prev).set(newGenome.id, [visualNode]));

        setConfigPanelOpen(false);
        setConfigNodeType(null);
        setEditingNodeId(null);
    }, [networkState]);

    const openConfigPanel = useCallback((type: string, nodeId?: string) => {
        setConfigNodeType(type);
        setEditingNodeId(nodeId || null);
        setConfigPanelOpen(true);
      }, []);

    return (
        <div className={styles.container}>
            <TitleBar />

            <div className={styles.content}>
                <SideMenu handleAddNode={openConfigPanel}/>

                <NetworkCanvas
                    onNodeSelect={setSelectedNode}
                    onGenomeSelect={setSelectedGenome}
                    genomesState={genomesState}
                    networkState={networkState}
                    configPanelOpen={configPanelOpen}
                    setConfigPanelOpen={setConfigPanelOpen}
                    configNodeType={configNodeType}
                    setConfigNodeType={setConfigNodeType}
                    editingNodeId={editingNodeId}
                    setEditingNodeId={setEditingNodeId}
                    openConfigPanel={openConfigPanel}
                />

                <SidePanel
                    selectedNode={selectedNode}
                    genomes={Array.from(genomesState[0].values())}
                />
            </div>
        </div>
    );
}