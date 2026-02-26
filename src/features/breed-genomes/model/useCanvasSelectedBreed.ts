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

            let breedResult;

            // Determine active strategy (randomly picking one of the selected ones, default to subgraph-insertion)
            const activeStrategies = settings.selectedCrossovers.filter(s =>
                s === 'subgraph-insertion' || s === 'subgraph-replacement' || s === 'neat-style' || s === 'multi-point'
            );

            const chosenStrategy = activeStrategies.length > 0
                ? activeStrategies[Math.floor(Math.random() * activeStrategies.length)]
                : 'subgraph-insertion';

            if (chosenStrategy === 'subgraph-replacement') {
                breedResult = fromGenome.genome.BreedByReplacement(toGenome.genome, maxNodes);
            } else if (chosenStrategy === 'neat-style') {
                breedResult = fromGenome.genome.BreedNeatStyle(toGenome.genome, maxNodes);
            } else if (chosenStrategy === 'multi-point') {
                breedResult = fromGenome.genome.BreedMultiPoint(toGenome.genome, maxNodes);
            } else {
                breedResult = fromGenome.genome.Breed(toGenome.genome, maxNodes);
            }

            if (!breedResult) {
                alert("Breed failed: exceeded node limit or no valid insertion point found.");
                setBreedingStartGenomeId(null);
                return;
            }
            addGenome(breedResult.nodes, breedResult.genome, canvasWidth, canvasHeight, translate.x, translate.y, scale, 300);
            setBreedingStartGenomeId(null);
        } else if (!breedingStartGenomeId) {
            setBreedingStartGenomeId(genomeId);
        } else {
            setBreedingStartGenomeId(null);
        }
    }

}