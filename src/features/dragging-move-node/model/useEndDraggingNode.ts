import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useEndDraggingNode = () => {
    const setDraggingNodeId = useCanvasStateStore(state => state.setDraggingNodeId);
    
    return () => setDraggingNodeId(null);
}