import { Genome, serializeGenome } from "../../../entities/canvas-genome"
import { saveGenomeApi } from "../api/saveGenome";

export const useSaveGenome = () => {
    return async (genome: Genome) => {
        const genomeStr = await serializeGenome(genome);
        await saveGenomeApi(genomeStr);
    }
}