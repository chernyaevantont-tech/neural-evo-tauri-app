import { 
    Genome, 
    BaseNode, 
    Conv2DNode, 
    DenseNode, 
    FlattenNode, 
    InputNode, 
    OutputNode,
    PoolingNode,
    AddNode,
    Concat2DNode
} from "..";

export type ConnectionIndexes = {fromIndex: number, toIndex: number}[];

export const deserializeGenome = async (genomeStr: string): Promise<{
    nodes: BaseNode[], 
    genome: Genome, 
}> => {
    const rows = genomeStr.split("\n");

    const nodes: BaseNode[] = [];

    let rowIndex = 0;
    while (rows[rowIndex] != "CONNECTIONS" && rowIndex < rows.length) {
        const obj = JSON.parse(rows[rowIndex]);
        rowIndex++;

        switch (obj.node) {
            case "Input":
                const outputShape = obj.params.output_shape;
                nodes.push(new InputNode(outputShape));
                break;
            case "Dense":
                {
                    const units = obj.params.units;
                    const activation = obj.params.activation;
                    const use_bias = obj.params.use_bias;
                    nodes.push(new DenseNode(units, activation, use_bias));
                    break;
                }
            case "Conv2D":
                {
                    const filters = obj.params.filters;
                    const kernel_size = obj.params.kernel_size;
                    const stride = obj.params.stride;
                    const padding = obj.params.padding;
                    const dilation = obj.params.dilation;
                    const use_bias = obj.params.use_bias;
                    nodes.push(new Conv2DNode(filters, kernel_size, stride, padding, dilation, use_bias));
                    break;
                }
            case "Pooling":
                {
                    const pool_type = obj.params.pool_type;
                    const kernel_size = obj.params.kernel_size;
                    const stride = obj.params.stride;
                    const padding = obj.params.padding;
                    nodes.push(new PoolingNode(pool_type, kernel_size, stride, padding));
                    break;
                }
            case "Flatten":
                {
                    nodes.push(new FlattenNode());
                    break;
                }
            case "Add":
                {
                    nodes.push(new AddNode());
                    break;
                }
            case "Concat":
                {
                    nodes.push(new Concat2DNode());
                    break;
                }
            case "Output":
                {
                    const inputShape = obj.params.input_shape;
                    nodes.push(new OutputNode(inputShape));
                    break;
                }
            default:
                {
                    throw new Error("Wrong node type");
                }
        }
    }

    rowIndex++;

    for (; rowIndex < rows.length - 1; rowIndex++) {
        const indexes = rows[rowIndex].split(" ");
        if (indexes.length != 2) {
            throw new Error("Wrong connections row");
        }   
        const fromNodeIndex = Number.parseInt(indexes[0]);
        const toNodeIndex = Number.parseInt(indexes[1]);
        
        nodes[fromNodeIndex].AddNext(nodes[toNodeIndex]);
    }

    const inputNodes: BaseNode[] = [];
    const outputNodes: BaseNode[] = []; 
    for (let node of nodes) {
        if (node.previous.length == 0) {
            inputNodes.push(node);
        }
        if (node.next.length == 0) {
            outputNodes.push(node);
        }
    }

    return {
        nodes: nodes,
        genome: new Genome(inputNodes, outputNodes),
    };
}