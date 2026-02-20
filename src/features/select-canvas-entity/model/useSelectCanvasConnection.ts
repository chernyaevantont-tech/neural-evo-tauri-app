import { useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { useCanvasStateStore } from "../../../entities/canvas-state";

export const useSelectCanvasConnection = () => {
    const connections = useCanvasGenomeStore(state => state.connections);
    const nodes = useCanvasGenomeStore(state => state.nodes);
    const setSelectedNodeId = useCanvasStateStore(state => state.setSelectedNodeId);
    const setSelectedGenomeId = useCanvasStateStore(state => state.setSelectedGenomeId);
    const setSelectedConnectionId = useCanvasStateStore(state => state.setSelectedConnectionId);

    return (connectionId: string) => {
        const connection = connections.get(connectionId);
        if (!connection) return;

        const genomeId = nodes.get(connection.fromNodeId)!.genomeId;
        
        setSelectedNodeId(null);
        setSelectedGenomeId(genomeId);
        setSelectedConnectionId(connectionId);
    }
}