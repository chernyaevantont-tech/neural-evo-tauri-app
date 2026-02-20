import { useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useDeleteNode = () => {
    const deleteNode = useCanvasGenomeStore(state => state.deleteNode);

    return (nodeId: string) => {
        deleteNode(nodeId);
    }
}