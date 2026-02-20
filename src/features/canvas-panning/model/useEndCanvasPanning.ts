import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useEndCanvasPanning = () => {
    const setIsPanning = useCanvasStateStore(state => state.setIsPanning)

    return () => setIsPanning(false);
}