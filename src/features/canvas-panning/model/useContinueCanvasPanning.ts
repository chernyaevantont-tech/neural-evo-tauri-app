import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useContinueCanvasPanning = () => {
    const isPanning = useCanvasStateStore(state => state.isPanning);
    const lastPanningPos = useCanvasStateStore(state => state.lastPanningPos);
    const setLastPanningPos = useCanvasStateStore(state => state.setLastPanningPos);
    const translate = useCanvasStateStore(state => state.translate);
    const setTranslate = useCanvasStateStore(state => state.setTranslate);
    
    return (clientX: number, clientY: number) => {
        if(!isPanning) return;

        const dx = clientX - lastPanningPos.x;
        const dy = clientY - lastPanningPos.y;

        setTranslate({x: translate.x + dx, y: translate.y + dy});
        setLastPanningPos({x: clientX, y: clientY});
    }
}