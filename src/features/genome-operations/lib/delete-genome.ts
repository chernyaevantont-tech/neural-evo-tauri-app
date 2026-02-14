import { Dispatch, SetStateAction } from "react";
import { Connection, VisualGenome, VisualNode } from "../../../components/types";

export const deleteGenome = (
    deletingGenomeId: string,
    setGenomes: Dispatch<SetStateAction<Map<string, VisualGenome>>>,
    nodes: Map<string, VisualNode>,
    setNodes: Dispatch<SetStateAction<Map<string, VisualNode>>>,
    genomeNode: Map<string, VisualNode[]>,
    setGenomeNode: Dispatch<SetStateAction<Map<string, VisualNode[]>>>,
    setConnections: Dispatch<SetStateAction<Map<string, Connection>>>
) => {
    const nodesToDelete = genomeNode.get(deletingGenomeId)!;
    const newNodes = new Map(nodes);
    nodesToDelete.forEach(node => newNodes.delete(node.node.id));
    setConnections(prev => {
        const newConnections = new Map(prev);
        for (const [id, conn] of prev) {
            if (!newNodes.has(conn.fromNodeId)) {
                newConnections.delete(id);
            }
        }
        return newConnections;
    });
    setGenomes(prev => {
        const newGenomes = new Map(prev);
        newGenomes.delete(deletingGenomeId);
        return newGenomes;
    });
    setNodes(newNodes);
    setGenomeNode(prev => {
        const newGenomeNode = new Map(prev);
        newGenomeNode.delete(deletingGenomeId);
        return newGenomeNode;
    });
}