import { useCanvasGenomeStore } from "../../../entities/canvas-genome";
import { useCanvasStateStore } from "../../../entities/canvas-state";

export const useStartDraggingNode = () => {
    const setDraggingNodeId = useCanvasStateStore(state => state.setDraggingNodeId);
    const setDragOffset = useCanvasStateStore(state => state.setDragOffset);
    const setNodeContextMenu = useCanvasStateStore(state => state.setNodeContextMenu);
    const setConnectionContextMenu = useCanvasStateStore(state => state.setConnectionContextMenu);
    const setGraphContextMenu = useCanvasStateStore(state => state.setGenomeContextMenu);
    const translate = useCanvasStateStore(state => state.translate);
    const scale = useCanvasStateStore(state => state.scale);
    const nodes = useCanvasGenomeStore(state => state.nodes);

    return (nodeId: string, canvasLeft: number, canvasTop: number, clientX: number, clientY: number) => {
        const node = nodes.get(nodeId);
        if (!node) return;

        setNodeContextMenu(null);
        setConnectionContextMenu(null);
        setGraphContextMenu(null);


        const mouseX = clientX - canvasLeft;
        const mouseY = clientY - canvasTop;
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        setDragOffset({
            x: worldX - node.position.x,
            y: worldY - node.position.y,
        });
        setDraggingNodeId(nodeId);
    }
}