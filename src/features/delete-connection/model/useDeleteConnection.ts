import { useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useDeleteConnection = () => {
    const deleteConnection = useCanvasGenomeStore(state => state.deleteConnection);

    return (connectionId: string) => {
        deleteConnection(connectionId);
    }
}