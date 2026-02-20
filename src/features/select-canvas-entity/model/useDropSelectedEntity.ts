import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useDropSelectedEntity = () => {
    const setSelectedNodeId = useCanvasStateStore(state => state.setSelectedNodeId);
    const setSelectedGenomeId = useCanvasStateStore(state => state.setSelectedGenomeId);
    const setSelectedConnectionId = useCanvasStateStore(state => state.setSelectedConnectionId);

    return () => {
        setSelectedNodeId(null);
        setSelectedGenomeId(null);
        setSelectedConnectionId(null);
    }
}