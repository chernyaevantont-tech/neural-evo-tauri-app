import { useCanvasGenomeStore } from "../../../entities/canvas-genome"

export const useCopyNode = () => {
    const addNode = useCanvasGenomeStore(state => state.addNode);
    const nodes = useCanvasGenomeStore(state => state.nodes);

    return (nodeId: string) => {
        const node = nodes.get(nodeId);
        if (!node) return;

        addNode(
            node.node.Clone(),
            {x: node.position.x + 10, y: node.position.y - 10}
        );
    }
}