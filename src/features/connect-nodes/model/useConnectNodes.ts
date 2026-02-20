import { useCanvasGenomeStore } from "../../../entities/canvas-genome";
import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useConnectNodes = () => {
    const setConnectingFromNodeId = useCanvasStateStore(state => state.setConnectingFromNodeId);
    const connectingFromNodeId = useCanvasStateStore(state => state.connectingFromNodeId);
    const connectNodes = useCanvasGenomeStore(state => state.connectNodes);

    return (nodeId: string | null) => {
        if (nodeId && connectingFromNodeId && nodeId != connectingFromNodeId) {
            connectNodes(connectingFromNodeId, nodeId);
            setConnectingFromNodeId(null);
        } else if (!connectingFromNodeId) {
            setConnectingFromNodeId(nodeId);
        } else {
            setConnectingFromNodeId(null);
        }
    }
}