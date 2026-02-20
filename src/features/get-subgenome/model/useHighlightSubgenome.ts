import { useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useHighlightSubgenome = () => {
    const highlightNodes = useCanvasGenomeStore(state => state.highlightNodes);
    const genomes = useCanvasGenomeStore(state => state.genomes);
    const genomeNode = useCanvasGenomeStore(state => state.genomeNode);

    return (genomeId: string) => {
        const genome = genomes.get(genomeId);
        const genomeNodeIds = genomeNode.get(genomeId)?.map(n => n.node.id);
        if (!genome || !genomeNodeIds) return;

        const subgenomeNodeIds = genome.genome.GetRandomSubgenomeNodeIds();

        const subgenomeNodeIdsSet = new Set<string>(subgenomeNodeIds); 
        const hightlightedNodes: {nodeId: string, isHighlighted: boolean}[] = [];
        genomeNodeIds.forEach(id => {
            if (subgenomeNodeIdsSet.has(id)) {
                hightlightedNodes.push({nodeId: id, isHighlighted: true});
            } else {
                hightlightedNodes.push({nodeId: id, isHighlighted: false});
            }
        })
        highlightNodes(hightlightedNodes);
    }
}