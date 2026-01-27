import { BaseNode } from "./nodes/base_node"

export class Genome {
    private inputNodes: BaseNode[]
    private outputNodes: BaseNode[]

    constructor(
        inputNodes: BaseNode[],
        outputNodes: BaseNode[]
    ) {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
    }

    public GetRandomSubgraph(): Genome {
        const startNodes = this.inputNodes.map(x => x.next).flat().filter(x => x.next.length > 0)

        let subgraphStartNode: BaseNode | null = null;
        let subgraphEndNode: BaseNode | null = null;

        let currentNodes: BaseNode[] = []

        while (true) {
            if (subgraphStartNode == null) {
                currentNodes = [...startNodes]
            } else {
                currentNodes = subgraphStartNode.next.filter(x => x.next.length > 0)
                if (currentNodes.length == 0) {
                    subgraphEndNode = subgraphStartNode
                    break
                }
            }
            while (currentNodes.length != 0) {
                currentNodes.concat(currentNodes[0].next.filter(x => x.next.length > 0))
                if (Math.random() < 0.2) {
                    if (subgraphStartNode == null) {
                        subgraphStartNode = currentNodes[0]
                        currentNodes = subgraphStartNode.next.filter(x => x.next.length > 0)
                    }
                    else {
                        subgraphEndNode = currentNodes[0]
                        break
                    }
                }
                else {
                    currentNodes.shift()
                }
            }
            
            if (subgraphEndNode != null) {
                break
            }
        }

        return new Genome([subgraphStartNode!], [subgraphEndNode!])
    }
}