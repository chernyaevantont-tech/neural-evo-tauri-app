import { useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { useCanvasStateStore } from "../../../entities/canvas-state";

export const useContinueMovingNode = () => {
    const moveNodes = useCanvasGenomeStore(state => state.moveNodes);
    const draggingNodeId = useCanvasStateStore(state =>  state.draggingNodeId);
    const translate = useCanvasStateStore(state => state.translate);
    const scale = useCanvasStateStore(state => state.scale);
    const dragOffset = useCanvasStateStore (state => state.dragOffset);

    return (canvasLeft: number, canvasTop: number, clientX: number, clientY: number) => {
        if (!draggingNodeId) return;

        const mouseX = clientX - canvasLeft;
        const mouseY = clientY - canvasTop;
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        const position = {
            x: Math.round(worldX - dragOffset.x),
            y: Math.round(worldY - dragOffset.y),
        };

        moveNodes([{nodeId: draggingNodeId, position}]);
    }
}