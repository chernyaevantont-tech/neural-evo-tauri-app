import { useCallback, useRef, useState } from 'react';
import styles from './NetworkEditorPage.module.css';
import { VisualGenome, VisualNode } from '../../components/types';
import { NetworkCanvas, SidePanel, TitleBar } from '../../widgets';
import { MenuType, SideMenu } from '../../widgets/side-menu/SideMenu';
import { useCanvasInteraction, useNetworkState } from './hooks';
import { BaseNode } from '../../evo/nodes/base_node';
import { Position } from '../../shared/types';
import { loadGenomeFromFile } from '../../shared/api';
import { v4 } from 'uuid';
import { Genome } from '../../evo/genome';
import { addNewGenome } from '../../features/genome-operations/lib/add-new-genome';

export const NetworkEditorPage: React.FC = () => {
    const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
    const [selectedGenome, setSelectedGenome] = useState<VisualGenome | null>(null);

    const networkState = useNetworkState();
    const canvasState = useCanvasInteraction();

    const [configPanelOpen, setConfigPanelOpen] = useState<boolean>(false);
    const [configNodeType, setConfigNodeType] = useState<string | null>(null);
    const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    const [menuType, setMenuType] = useState<MenuType>("Layers");

    const openConfigPanel = useCallback((type: string, nodeId?: string) => {
        setConfigNodeType(type);
        setEditingNodeId(nodeId || null);
        setConfigPanelOpen(true);
    }, []);

    const handleGetSubgenome = () => {
        if (!networkState.selectedGenomeId) return;

        const newNodes = new Map(networkState.nodes);

        const currentGenomeNodes = networkState.genomeNode.get(networkState.selectedGenomeId)!;

        // Reset highlighting for all nodes in the genome
        for (let node of currentGenomeNodes) {
            const nodeId = node.node.id;
            const existingNode = newNodes.get(nodeId);
            if (existingNode) {
                newNodes.set(nodeId, { ...existingNode, highlighted: false });
            }
        }

        const subGenomeNodeIds = networkState.genomes.get(networkState.selectedGenomeId)!.genome.GetRandomSubgenomeNodeIds();
        for (let nodeId of subGenomeNodeIds) {
            const existingNode = newNodes.get(nodeId);
            if (existingNode) {
                newNodes.set(nodeId, { ...existingNode, highlighted: true });
            }
        }

        networkState.setNodes(newNodes);

        // Sync genomeNode state with new node objects
        const newGenomeNode = new Map(networkState.genomeNode);
        const updatedGenomeNodes = currentGenomeNodes
            .map(node => newNodes.get(node.node.id))
            .filter((n): n is VisualNode => n !== undefined);

        newGenomeNode.set(networkState.selectedGenomeId, updatedGenomeNodes);
        networkState.setGenomeNode(newGenomeNode);
    };


    const handleLoadGenome = () => {
        loadGenomeFromFile((data: ReturnType<typeof import('../../saver/loadGenome').loadGenome>) => {
            if (svgRef.current) {
                const svg = svgRef.current;
                const rect = svg.getBoundingClientRect();
                addNewGenome(
                    data.nodes,
                    data.genome,
                    data.isValid,
                    networkState.nodes,
                    networkState.setNodes,
                    networkState.setGenomes,
                    networkState.setGenomeNode,
                    networkState.setConnections,
                    rect.width,
                    rect.height,
                    canvasState.translate.x,
                    canvasState.translate.y,
                    canvasState.scale,
                )
            }
        });
    };

    return (
        <div className={styles.container}>
            <TitleBar />

            <div className={styles.content}>
                <SideMenu
                    handleAddNode={openConfigPanel}
                    handleLoadGenome={handleLoadGenome}
                    handleGetSubgenome={handleGetSubgenome}
                    menuType={menuType}
                    setMenuType={(value) => { setMenuType(value); canvasState.setConnectingFrom(null) }}
                />

                <NetworkCanvas
                    onNodeSelect={setSelectedNode}
                    onGenomeSelect={setSelectedGenome}
                    genomesState={[networkState.genomes, networkState.setGenomes]}
                    networkState={networkState}
                    canvasState={canvasState}
                    configPanelOpen={configPanelOpen}
                    setConfigPanelOpen={setConfigPanelOpen}
                    configNodeType={configNodeType}
                    setConfigNodeType={setConfigNodeType}
                    editingNodeId={editingNodeId}
                    setEditingNodeId={setEditingNodeId}
                    openConfigPanel={openConfigPanel}
                    svgRef={svgRef}
                    menuType={menuType}
                />

                <SidePanel
                    selectedNode={selectedNode}
                    genomes={Array.from(networkState.genomes.values())}
                    menuType={menuType}
                />
            </div>
        </div>
    );
}