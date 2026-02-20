import { deserializeGenome, useCanvasGenomeStore } from "../../../entities/canvas-genome"
import { useCanvasStateStore } from "../../../entities/canvas-state";
import { loadGenomeApi } from "../api/loadGenome"

export const useLoadGenome = () => {
    const addNewGenome = useCanvasGenomeStore(state => state.addGenome);
    const translate = useCanvasStateStore(state => state.translate);
    const scale = useCanvasStateStore(state => state.scale);
    const canvasWidth = useCanvasStateStore(state => state.canvasWidth);
    const canvasHeight = useCanvasStateStore(state => state.canvasHeight);

    return async () => {
        const genomeStr = await loadGenomeApi();
        const genome = await deserializeGenome(genomeStr);
        addNewGenome(
            genome.nodes,
            genome.genome,
            canvasWidth,
            canvasHeight,
            translate.x,
            translate.y,
            scale,
            300
        );
    }
}