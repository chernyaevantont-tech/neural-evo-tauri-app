import { useCanvasStateStore } from "../../../entities/canvas-state";

export const useStartCanvasPanning = () => {
    const setIsPanning = useCanvasStateStore(state => state.setIsPanning);
    const setLastPanningPos = useCanvasStateStore(state => state.setLastPanningPos);
    const setNodeContextMenu = useCanvasStateStore(state => state.setNodeContextMenu);
    const setConnectionContextMenu = useCanvasStateStore(state => state.setConnectionContextMenu);
    const setGraphContextMenu = useCanvasStateStore(state => state.setGenomeContextMenu);

    return (clientX: number, clientY: number) => {
        setIsPanning(true);
        setLastPanningPos({x: clientX, y: clientY});
        setNodeContextMenu(null);
        setConnectionContextMenu(null);
        setGraphContextMenu(null);
    }
}