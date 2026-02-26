import { useCanvasGenomeStore } from "../../../entities/canvas-genome";
import { useCanvasStateStore } from "../../../entities/canvas-state"
import { useEvolutionSettingsStore } from "../../evolution-manager";

export const useCanvasSelectedBreed = () => {
    const breedingStartGenomeId = useCanvasStateStore(state => state.breedingStartGenomeId);
    const setBreedingStartGenomeId = useCanvasStateStore(state => state.setBreedingStartGenomeId)
    const addGenome = useCanvasGenomeStore(state => state.addGenome);
    const genomes = useCanvasGenomeStore(state => state.genomes);
    const canvasWidth = useCanvasStateStore(state => state.canvasWidth);
    const canvasHeight = useCanvasStateStore(state => state.canvasHeight);
    const translate = useCanvasStateStore(state => state.translate);
    const scale = useCanvasStateStore(state => state.scale);

    const settings = useEvolutionSettingsStore();

    return (genomeId: string | null) => {
        if (genomeId && breedingStartGenomeId && genomeId != breedingStartGenomeId) {
            const fromGenome = genomes.get(breedingStartGenomeId);
            const toGenome = genomes.get(genomeId);
            if (!fromGenome || !toGenome) return;

            const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

            const newGenome = fromGenome.genome.Breed(toGenome.genome, maxNodes);
            if (!newGenome) {
                alert("Breed failed: exceeded node limit or no valid insertion point found.");
                setBreedingStartGenomeId(null);
                return;
            }
            addGenome(newGenome.nodes, newGenome.genome, canvasWidth, canvasHeight, translate.x, translate.y, scale, 300);
            setBreedingStartGenomeId(null);
        } else if (!breedingStartGenomeId) {
            setBreedingStartGenomeId(genomeId);
        } else {
            setBreedingStartGenomeId(null);
        }
    }

}