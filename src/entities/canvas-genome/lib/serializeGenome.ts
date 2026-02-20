import { Genome, BaseNode } from "..";

export const serializeGenome = async (genome: Genome): Promise<string> => {
    const nodes: BaseNode[] = [];
    const nodeIndexes: Map<BaseNode, number> = new Map();
    const nodesToCheck: BaseNode[] = [...genome.inputNodes];

    let output: string = "";
    let nodeCounter = 0;
    
    while (nodesToCheck.length > 0) {
        const currentNode = nodesToCheck.shift()!;
        if (nodeIndexes.get(currentNode) != undefined) continue;

        nodes.push(currentNode);
        nodeIndexes.set(currentNode, nodeCounter);
        nodeCounter++
        nodesToCheck.push(...currentNode.previous, ...currentNode.next);
        
        output += currentNode.GetInfo() + "\n";
    }

    output += "CONNECTIONS\n";

    for (let node of nodes) {
        const currentIndex = nodeIndexes.get(node);
        for (let nextNode of node.next) {
            output += `${currentIndex} ${nodeIndexes.get(nextNode)!}\n`
        }
    }

    return output;
}
