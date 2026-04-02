// Original type definitions
export type ActivationFunction = "relu" | "leaky_relu" | "softmax" | "sigmoid" | "tanh" | "linear" | "gelu" | "swish"
export type KernelSize = {h: number, w: number}
export type PoolType = "max" | "avg"

/**
 * NodeType Enum
 *
 * Type-safe identifiers for neural network layer types.
 * Used for serialization, deserialization, and type checking.
 */
export enum NodeType {
    // Core layers
    Input = 'Input',
    Dense = 'Dense',
    Conv2D = 'Conv2D',
    Conv1D = 'Conv1D',
    Pooling = 'Pooling',
    Flatten = 'Flatten',
    Output = 'Output',

    // Normalization layers
    BatchNorm = 'BatchNorm',
    LayerNorm = 'LayerNorm',

    // Regularization layers
    Dropout = 'Dropout',
    Dropout2D = 'Dropout2D',
    GaussianNoise = 'GaussianNoise',

    // Merge layers
    Add = 'Add',
    Concat = 'Concat',

    // Attention layers
    MultiHeadAttention = 'MultiHeadAttention',
    TransformerEncoderBlock = 'TransformerEncoderBlock',

    // Sequential layers
    LSTM = 'LSTM',
    GRU = 'GRU',
}

/**
 * Type guard to check if a string is a valid NodeType
 */
export function isValidNodeType(value: string): value is NodeType {
    return Object.values(NodeType).includes(value as NodeType);
}

/**
 * Get NodeType from string, returns undefined if invalid
 */
export function parseNodeType(value: string): NodeType | undefined {
    return isValidNodeType(value) ? value as NodeType : undefined;
}
