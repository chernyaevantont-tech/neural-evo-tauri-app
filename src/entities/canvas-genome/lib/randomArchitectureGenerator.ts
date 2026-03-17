import {
    Genome,
    BaseNode,
    InputNode,
    OutputNode,
    DenseNode,
    Conv2DNode,
    Conv1DNode,
    LSTMNode,
    GRUNode,
    PoolingNode,
    FlattenNode,
    DropoutNode,
    BatchNormNode,
    LayerNormNode
} from "..";
import type { KernelSize, ActivationFunction, PoolType } from "../model/nodes/types";
import { RandomizeInteger } from "../../../lib/random";

// Helper functions for randomness
const randomInt = (min: number, max: number): number => RandomizeInteger(min, max);
const randomChoice = <T,>(array: T[]): T => array[randomInt(0, array.length - 1)];

/**
 * Generates a random neural network architecture based on input and output shapes.
 * Uses the input shape to determine the appropriate layer types.
 */
export const generateRandomArchitecture = (
    inputShape: number[],
    outputShape: number[],
    options?: {
        maxDepth?: number;
        useAttention?: boolean;
    }
): Genome => {
    const maxDepth = options?.maxDepth ?? 8;
    const useAttention = options?.useAttention ?? false;

    const nodes: BaseNode[] = [];

    // Infer data type from input shape
    const inputDimensions = inputShape.length;
    const inputNode = new InputNode(inputShape);
    nodes.push(inputNode);

    let currentNode: BaseNode = inputNode;
    const depth = randomInt(2, maxDepth);

    if (inputDimensions === 3) {
        // Image data: [height, width, channels]
        currentNode = buildConvolutionalPath(nodes, currentNode, depth);
    } else if (inputDimensions === 2) {
        // Sequence or matrix data: [seq_len, features]
        currentNode = buildSequentialPath(nodes, currentNode, depth, useAttention);
    } else if (inputDimensions === 1) {
        // Dense vector: [features]
        currentNode = buildDensePath(nodes, currentNode, depth);
    }

    // Final layers: adapt to output shape
    currentNode = adaptToOutputShape(nodes, currentNode, outputShape);

    // Add output node
    const outputNode = new OutputNode(outputShape);
    currentNode.AddNext(outputNode);
    nodes.push(outputNode);

    return new Genome([inputNode], [outputNode]);
};

/**
 * Build a convolutional path for image data
 */
const buildConvolutionalPath = (
    nodes: BaseNode[],
    currentNode: BaseNode,
    depth: number
): BaseNode => {
    let currentDepth = 0;

    while (currentDepth < depth && currentDepth < 4) {
        const choice = randomInt(0, 100);

        if (choice < 40) {
            // Add Conv2D layer
            const filters = randomChoice([16, 32, 64, 128]);
            const kernelSize: KernelSize = { h: 3, w: 3 };
            const stride = randomChoice([1, 1, 1, 2]); // Bias towards stride 1
            const padding = 1; // Same padding
            const activation = randomChoice(['relu', 'relu', 'leaky_relu'] as const);

            const conv = new Conv2DNode(filters, kernelSize, stride, padding, 1, true, activation);
            currentNode.AddNext(conv);
            nodes.push(conv);
            currentNode = conv;
        } else if (choice < 70) {
            // Add Pooling layer
            const poolType = randomChoice(['max', 'avg'] as const);
            const kernelSize: KernelSize = { h: 2, w: 2 };
            const stride = 2;

            const pool = new PoolingNode(poolType, kernelSize, stride, 0);
            currentNode.AddNext(pool);
            nodes.push(pool);
            currentNode = pool;
        } else if (choice < 85) {
            // Add BatchNorm
            const bn = new BatchNormNode(1e-5, 0.1);
            currentNode.AddNext(bn);
            nodes.push(bn);
            currentNode = bn;
        } else {
            // Add Dropout
            const dropout = new DropoutNode(randomChoice([0.2, 0.25, 0.3]));
            currentNode.AddNext(dropout);
            nodes.push(dropout);
            currentNode = dropout;
        }

        currentDepth++;
    }

    // Flatten for transition to dense layers
    const flatten = new FlattenNode();
    currentNode.AddNext(flatten);
    nodes.push(flatten);
    currentNode = flatten;

    return currentNode;
};

/**
 * Build a sequential/temporal path for sequence data
 */
const buildSequentialPath = (
    nodes: BaseNode[],
    currentNode: BaseNode,
    depth: number,
    useAttention: boolean
): BaseNode => {
    let currentDepth = 0;

    while (currentDepth < depth && currentDepth < 3) {
        const choice = randomInt(0, 100);

        if (choice < 35) {
            // Add Conv1D layer
            const filters = randomChoice([32, 64, 128]);
            const kernelSize = randomChoice([3, 5, 7]);
            const stride = 1;
            const padding = 1;
            const activation = randomChoice(['relu', 'relu', 'tanh'] as const);

            const conv1d = new Conv1DNode(filters, kernelSize, stride, padding, 1, true, activation);
            currentNode.AddNext(conv1d);
            nodes.push(conv1d);
            currentNode = conv1d;
        } else if (choice < 50) {
            // Add LSTM layer
            const units = randomChoice([64, 128, 256]);
            const recurrentActivation = randomChoice(['sigmoid', 'tanh'] as const);
            const activation = randomChoice(['tanh', 'relu'] as const);

            const lstm = new LSTMNode(units, 'sigmoid', activation, recurrentActivation, true);
            currentNode.AddNext(lstm);
            nodes.push(lstm);
            currentNode = lstm;
        } else if (choice < 65) {
            // Add GRU layer
            const units = randomChoice([64, 128, 256]);
            const activation = randomChoice(['tanh', 'relu'] as const);

            const gru = new GRUNode(units, 'sigmoid', activation, true, true);
            currentNode.AddNext(gru);
            nodes.push(gru);
            currentNode = gru;
        } else if (choice < 80) {
            // Add LayerNorm
            const ln = new LayerNormNode(1e-5);
            currentNode.AddNext(ln);
            nodes.push(ln);
            currentNode = ln;
        } else {
            // Add Dropout
            const dropout = new DropoutNode(randomChoice([0.2, 0.3]));
            currentNode.AddNext(dropout);
            nodes.push(dropout);
            currentNode = dropout;
        }

        currentDepth++;
    }

    return currentNode;
};

/**
 * Build a dense path for vector data
 */
const buildDensePath = (
    nodes: BaseNode[],
    currentNode: BaseNode,
    depth: number
): BaseNode => {
    let currentDepth = 0;

    while (currentDepth < depth && currentDepth < 6) {
        const choice = randomInt(0, 100);

        if (choice < 60) {
            // Add Dense layer
            const units = randomChoice([32, 64, 128, 256]);
            const activation = randomChoice(['relu', 'relu', 'leaky_relu', 'tanh'] as const);

            const dense = new DenseNode(units, activation, true);
            currentNode.AddNext(dense);
            nodes.push(dense);
            currentNode = dense;
        } else if (choice < 75) {
            // Add Dropout
            const dropout = new DropoutNode(randomChoice([0.2, 0.3, 0.4]));
            currentNode.AddNext(dropout);
            nodes.push(dropout);
            currentNode = dropout;
        } else {
            // Add BatchNorm
            const bn = new BatchNormNode(1e-5, 0.1);
            currentNode.AddNext(bn);
            nodes.push(bn);
            currentNode = bn;
        }

        currentDepth++;
    }

    return currentNode;
};

/**
 * Add dense layers to adapt from hidden features to output shape
 */
const adaptToOutputShape = (
    nodes: BaseNode[],
    currentNode: BaseNode,
    outputShape: number[]
): BaseNode => {
    // Add 1-2 intermediate dense layers if output shape is non-trivial
    const outputSize = outputShape.reduce((a, b) => a * b, 1);

    if (outputSize > 128) {
        // Larger output space: add intermediate layer
        const intermediate = randomChoice([128, 256, 512]);
        const dense1 = new DenseNode(intermediate, 'relu', true);
        currentNode.AddNext(dense1);
        nodes.push(dense1);
        currentNode = dense1;

        // Optional dropout
        if (Math.random() < 0.3) {
            const dropout = new DropoutNode(0.2);
            currentNode.AddNext(dropout);
            nodes.push(dropout);
            currentNode = dropout;
        }
    }

    // Final dense layer to match output size
    const activation = outputSize > 2 ? 'softmax' : 'sigmoid';
    const outputDense = new DenseNode(outputSize, activation, true);
    currentNode.AddNext(outputDense);
    nodes.push(outputDense);

    return outputDense;
};

/**
 * Helper to extract input and output shapes from dataset profile
 */
export const extractShapesFromDatasetProfile = (
    streams: Array<{ 
        role: 'Input' | 'Target' | 'Ignore'; 
        tensorShape: number[],
        numClasses?: number,
    }>
): { inputShape: number[]; outputShape: number[] } | null => {
    const inputStreams = streams.filter(s => s.role === 'Input');
    const targetStreams = streams.filter(s => s.role === 'Target');

    if (inputStreams.length === 0 || targetStreams.length === 0) {
        return null;
    }

    // For simplicity, use first input and target stream
    // In future: could support multi-input/output by concatenating shapes
    const inputShape = inputStreams[0].tensorShape;
    
    // For Target stream: outputShape should be [num_classes] for classification
    const numClasses = targetStreams[0].numClasses || 2; // Default to 2 if not specified
    const outputShape = [numClasses];

    return { inputShape, outputShape };
};
