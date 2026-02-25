import { serializeGenome, useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { testTrainGenomeAPI, testTrainOnImageFolderAPI } from "../api/test-train-genome";

export const useTestTrainGenome = () => {
    const genomes = useCanvasGenomeStore(state => state.genomes);
    return async (genomeId: string) => {
        const visualGenome = genomes.get(genomeId);
        if (!visualGenome || !visualGenome.isValid) return;
        const genomeStr = await serializeGenome(visualGenome.genome);
        console.log(genomeStr);
        await testTrainGenomeAPI(genomeStr);
    }
}

export const useTestTrainOnImageFolder = () => {
    const genomes = useCanvasGenomeStore(state => state.genomes);
    return async (genomeId: string) => {
        const visualGenome = genomes.get(genomeId);
        if (!visualGenome || !visualGenome.isValid) return;
        const genomeStr = await serializeGenome(visualGenome.genome);
        await testTrainOnImageFolderAPI(genomeStr);
    }
}