import { BaseNode, useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useEditNode = () => {
    const editNode = useCanvasGenomeStore(state => state.editNode);
    
    return (nodeId: string, node: BaseNode) => {
        editNode(nodeId, node);
    };
}