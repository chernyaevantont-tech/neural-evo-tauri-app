import { useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useDeleteGenome = () => {
    const deleteGenome = useCanvasGenomeStore(state => state.deleteGenome);

    return (genomeId: string) => {
        deleteGenome(genomeId);
    }
}