import { useCallback } from "react";
import { BaseNode, useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { useCanvasStateStore } from "../../../entities/canvas-state";

export const useAddNode = () => {
    const addNode = useCanvasGenomeStore(state => state.addNode);
    const dragOffset = useCanvasStateStore(state => state.dragOffset);

    return useCallback((node: BaseNode) => {
        addNode(node, dragOffset);
    }, [dragOffset]);
}