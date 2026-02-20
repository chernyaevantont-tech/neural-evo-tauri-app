import { useCanvasGenomeStore } from "../../../entities/canvas-genome";
import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useSelectCanvasNode = () => {
    const nodes = useCanvasGenomeStore(state => state.nodes);
    const setSelectedNodeId = useCanvasStateStore(state => state.setSelectedNodeId);
    const setSelectedGenomeId = useCanvasStateStore(state => state.setSelectedGenomeId);
    const setSelectedConnectionId = useCanvasStateStore(state => state.setSelectedConnectionId);

    return (nodeId: string) => {
        const node = nodes.get(nodeId);
        if (!node) return;
        
        setSelectedNodeId(nodeId);
        setSelectedGenomeId(node.genomeId);
        setSelectedConnectionId(null);
    }
}