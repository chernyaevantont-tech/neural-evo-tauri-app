import { serializeGenome, useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { testTrainGenomeAPI } from "../api/test-train-genome";

export const useTestTrainGenome = () => {
    const genomes = useCanvasGenomeStore(state => state.genomes);
    return async (genomeId: string) => {
        const visualGenome = genomes.get(genomeId);
        if (!visualGenome || !visualGenome.isValid) return;
        const genomeStr = await serializeGenome(visualGenome.genome);
        await testTrainGenomeAPI(genomeStr);
    }
}