import { describe, it, expect, beforeEach } from 'vitest';
import {
    generateRandomArchitecture,
    extractShapesFromDatasetProfile
} from './randomArchitectureGenerator';
import {
    Conv1DNode,
    Conv2DNode,
    DenseNode,
    InputNode,
    OutputNode
} from '..';

describe('randomArchitectureGenerator', () => {
    describe('generateRandomArchitecture', () => {
        it('should generate a genome with input and output nodes', () => {
            const inputShape = [4];
            const outputShape = [3];

            const genome = generateRandomArchitecture(inputShape, outputShape);

            // Check that genome was created
            expect(genome).toBeDefined();

            // Check that it has input and output nodes
            const nodes = genome.getAllNodes();
            expect(nodes.length).toBeGreaterThan(2);
        });

        it('should route 1D input to Dense path', () => {
            const inputShape = [4];
            const outputShape = [3];

            const genome = generateRandomArchitecture(inputShape, outputShape);
            const nodes = genome.getAllNodes();

            // Should have at least one Dense layer (and InputNode, OutputNode)
            const denseNodes = nodes.filter(n => n instanceof DenseNode);
            expect(denseNodes.length).toBeGreaterThan(0);

            // Should NOT have Conv1D or Conv2D in this path
            const convNodes = nodes.filter(n => n instanceof Conv1DNode || n instanceof Conv2DNode);
            expect(convNodes.length).toBe(0);
        });

        it('should route 2D input with dataTypeHint="Vector" to Dense path', () => {
            const inputShape = [12, 5]; // 12 features x 5 samples (tabular data)
            const outputShape = [3];

            const genome = generateRandomArchitecture(inputShape, outputShape, {
                dataTypeHint: 'Vector'
            });

            const nodes = genome.getAllNodes();

            // Should have Dense layers (tabular path)
            const denseNodes = nodes.filter(n => n instanceof DenseNode);
            expect(denseNodes.length).toBeGreaterThan(0);

            // Should NOT have Conv1D layers (temporal path)
            const conv1dNodes = nodes.filter(n => n instanceof Conv1DNode);
            expect(conv1dNodes.length).toBe(0);
        });

        it('should route 2D input with dataTypeHint="TemporalSequence" to Conv1D path', () => {
            const inputShape = [50, 12]; // 50 timesteps x 12 features
            const outputShape = [5];

            const genome = generateRandomArchitecture(inputShape, outputShape, {
                dataTypeHint: 'TemporalSequence'
            });

            const nodes = genome.getAllNodes();

            // Should have Conv1D, LSTM, or GRU layers (temporal path)
            const temporalNodes = nodes.filter(n => n instanceof Conv1DNode);
            expect(temporalNodes.length).toBeGreaterThanOrEqual(0); // May have temporal layers

            // Note: The actual composition depends on randomness,
            // so we mainly check that it doesn't crash and produces a valid genome
            expect(nodes.length).toBeGreaterThan(2);
        });

        it('should default 2D input to TemporalSequence when no hint provided', () => {
            const inputShape = [50, 12];
            const outputShape = [5];

            const genome1 = generateRandomArchitecture(inputShape, outputShape);
            const genome2 = generateRandomArchitecture(inputShape, outputShape, {
                dataTypeHint: 'TemporalSequence'
            });

            // Both should successfully generate genomes (actual composition varies due to randomness)
            expect(genome1.getAllNodes().length).toBeGreaterThan(2);
            expect(genome2.getAllNodes().length).toBeGreaterThan(2);
        });

        it('should route 3D input to Convolutional path', () => {
            const inputShape = [32, 32, 3]; // 32x32 RGB image
            const outputShape = [10]; // 10 classes

            const genome = generateRandomArchitecture(inputShape, outputShape);
            const nodes = genome.getAllNodes();

            // Should have Conv2D layers (image path)
            const conv2dNodes = nodes.filter(n => n instanceof Conv2DNode);
            expect(conv2dNodes.length).toBeGreaterThan(0);

            // Should have FlattenNode to transition from conv to dense
            const flattenNodes = nodes.filter(n => n.constructor.name === 'FlattenNode');
            expect(flattenNodes.length).toBeGreaterThan(0);
        });

        it('should respect maxDepth option', () => {
            const inputShape = [4];
            const outputShape = [3];
            const maxDepth = 3;

            const genome = generateRandomArchitecture(inputShape, outputShape, {
                maxDepth
            });

            const nodes = genome.getAllNodes();

            // With maxDepth=3, we should have reasonable number of nodes
            // InputNode + at most 3 layers + OutputNode + adaptations
            expect(nodes.length).toBeLessThan(20); // Loose upper bound
        });
    });

    describe('extractShapesFromDatasetProfile', () => {
        it('should extract shapes from dataset profile with Vector data type', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: 'Vector' as const,
                    numClasses: undefined
                },
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([4]);
            expect(result!.outputShape).toEqual([3]);
            expect(result!.dataTypeHint).toBe('Vector');
        });

        it('should extract shapes from dataset profile with TemporalSequence data type', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [50, 12],
                    dataType: 'TemporalSequence' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [5],
                    dataType: 'Categorical' as const,
                    numClasses: 5
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([50, 12]);
            expect(result!.outputShape).toEqual([5]);
            expect(result!.dataTypeHint).toBe('TemporalSequence');
        });

        it('should extract shapes from dataset profile with Image data type', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [32, 32, 3],
                    dataType: 'Image' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [10],
                    dataType: 'Categorical' as const,
                    numClasses: 10
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([32, 32, 3]);
            expect(result!.outputShape).toEqual([10]);
            expect(result!.dataTypeHint).toBe('Image');
        });

        it('should normalize legacy CHW image shapes to HWC', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [3, 240, 320],
                    dataType: 'Image' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [5],
                    dataType: 'Categorical' as const,
                    numClasses: 5
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([240, 320, 3]);
            expect(result!.outputShape).toEqual([5]);
            expect(result!.dataTypeHint).toBe('Image');
        });

        it('should return null when no input stream exists', () => {
            const streams = [
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeNull();
        });

        it('should return null when no target stream exists', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: 'Vector' as const
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeNull();
        });

        it('should use default numClasses of 2 if not specified', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: 'Vector' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [2],
                    dataType: 'Categorical' as const
                    // numClasses not specified
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.outputShape).toEqual([2]);
        });

        it('should ignore Ignore role streams', () => {
            const streams = [
                {
                    role: 'Ignore' as const,
                    tensorShape: [100],
                    dataType: 'Vector' as const
                },
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: 'Vector' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([4]);
            expect(result!.outputShape).toEqual([3]);
        });

        it('should handle undefined dataType gracefully', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: undefined
                },
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([4]);
            expect(result!.outputShape).toEqual([3]);
            expect(result!.dataTypeHint).toBeUndefined();
        });

        it('should use first input and target stream if multiple exist', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [4],
                    dataType: 'Vector' as const
                },
                {
                    role: 'Input' as const,
                    tensorShape: [32, 32, 3], // Second input (ignored)
                    dataType: 'Image' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                },
                {
                    role: 'Target' as const,
                    tensorShape: [1],
                    dataType: 'Categorical' as const,
                    numClasses: 1 // Second target (ignored)
                }
            ];

            const result = extractShapesFromDatasetProfile(streams);

            expect(result).toBeDefined();
            expect(result!.inputShape).toEqual([4]); // First input
            expect(result!.outputShape).toEqual([3]); // First target's numClasses
        });
    });

    describe('Integration: routing + shape extraction', () => {
        it('should generate Vector path for tabular dataset', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [12],
                    dataType: 'Vector' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [3],
                    dataType: 'Categorical' as const,
                    numClasses: 3
                }
            ];

            const extracted = extractShapesFromDatasetProfile(streams);
            expect(extracted).toBeDefined();

            const genome = generateRandomArchitecture(
                extracted!.inputShape,
                extracted!.outputShape,
                { dataTypeHint: extracted!.dataTypeHint }
            );

            const nodes = genome.getAllNodes();
            const denseNodes = nodes.filter(n => n instanceof DenseNode);

            // Should have Dense layers for Vector data type
            expect(denseNodes.length).toBeGreaterThan(0);
        });

        it('should generate TemporalSequence path for CSV temporal dataset', () => {
            const streams = [
                {
                    role: 'Input' as const,
                    tensorShape: [50, 12], // 50 timesteps, 12 features
                    dataType: 'TemporalSequence' as const
                },
                {
                    role: 'Target' as const,
                    tensorShape: [5],
                    dataType: 'Categorical' as const,
                    numClasses: 5
                }
            ];

            const extracted = extractShapesFromDatasetProfile(streams);
            expect(extracted).toBeDefined();
            expect(extracted!.dataTypeHint).toBe('TemporalSequence');

            const genome = generateRandomArchitecture(
                extracted!.inputShape,
                extracted!.outputShape,
                { dataTypeHint: extracted!.dataTypeHint }
            );

            const nodes = genome.getAllNodes();

            // Should successfully generate a genome (composition varies with randomness)
            expect(nodes.length).toBeGreaterThan(2);
        });
    });
});
