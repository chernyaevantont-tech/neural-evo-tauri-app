import {
    Genome,
    BaseNode,
    Conv2DNode,
    Conv1DNode,
    DenseNode,
    FlattenNode,
    InputNode,
    OutputNode,
    PoolingNode,
    LSTMNode,
    GRUNode,
    AddNode,
    Concat2DNode,
    MultiHeadAttentionNode,
    TransformerEncoderBlockNode,
    DropoutNode,
    BatchNormNode,
    LayerNormNode,
    Dropout2DNode,
    GaussianNoiseNode
} from "..";

export type ConnectionIndexes = { fromIndex: number, toIndex: number }[];

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
                    const activation = obj.params.activation || 'relu';
                    nodes.push(new Conv2DNode(filters, kernel_size, stride, padding, dilation, use_bias, activation));
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
            case "Dropout":
                {
                    const prob = obj.params.prob || 0.5;
                    nodes.push(new DropoutNode(prob));
                    break;
                }
            case "BatchNorm":
                {
                    const epsilon = obj.params.epsilon || 1e-5;
                    const momentum = obj.params.momentum || 0.1;
                    nodes.push(new BatchNormNode(epsilon, momentum));
                    break;
                }
            case "LayerNorm":
                {
                    const epsilon = obj.params.epsilon || 1e-5;
                    nodes.push(new LayerNormNode(epsilon));
                    break;
                }
            case "Dropout2D":
                {
                    const prob = obj.params.prob || 0.5;
                    nodes.push(new Dropout2DNode(prob));
                    break;
                }
            case "GaussianNoise":
                {
                    const std_dev = obj.params.std_dev || 0.1;
                    nodes.push(new GaussianNoiseNode(std_dev));
                    break;
                }
            case "Conv1D":
                {
                    const filters = obj.params.filters;
                    const kernel_size = obj.params.kernel_size;
                    const stride = obj.params.stride;
                    const padding = obj.params.padding;
                    const dilation = obj.params.dilation;
                    const use_bias = obj.params.use_bias;
                    const activation = obj.params.activation || 'relu';
                    nodes.push(new Conv1DNode(filters, kernel_size, stride, padding, dilation, use_bias, activation));
                    break;
                }
            case "LSTM":
                {
                    const hidden_units = obj.params.hidden_units;
                    const gate_activation = obj.params.gate_activation || 'sigmoid';
                    const cell_activation = obj.params.cell_activation || 'tanh';
                    const hidden_activation = obj.params.hidden_activation || 'tanh';
                    const use_bias = obj.params.use_bias !== undefined ? obj.params.use_bias : true;
                    nodes.push(new LSTMNode(hidden_units, gate_activation, cell_activation, hidden_activation, use_bias));
                    break;
                }
            case "GRU":
                {
                    const hidden_units = obj.params.hidden_units;
                    const gate_activation = obj.params.gate_activation || 'sigmoid';
                    const hidden_activation = obj.params.hidden_activation || 'tanh';
                    const use_bias = obj.params.use_bias !== undefined ? obj.params.use_bias : true;
                    const reset_after = obj.params.reset_after !== undefined ? obj.params.reset_after : true;
                    nodes.push(new GRUNode(hidden_units, gate_activation, hidden_activation, use_bias, reset_after));
                    break;
                }
            case "MultiHeadAttention":
                {
                    const n_heads = obj.params.n_heads;
                    const dropout = obj.params.dropout || 0.1;
                    const quiet_softmax = obj.params.quiet_softmax || false;
                    nodes.push(new MultiHeadAttentionNode(n_heads, dropout, quiet_softmax));
                    break;
                }
            case "TransformerEncoderBlock":
                {
                    const n_heads = obj.params.n_heads;
                    const d_ff = obj.params.d_ff;
                    const dropout = obj.params.dropout || 0.1;
                    const activation = obj.params.activation || 'relu';
                    const norm_first = obj.params.norm_first !== undefined ? obj.params.norm_first : true;
                    nodes.push(new TransformerEncoderBlockNode(n_heads, d_ff, dropout, activation, norm_first));
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