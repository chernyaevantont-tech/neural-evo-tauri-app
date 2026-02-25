import { describe, it, expect } from 'vitest';
import { Genome } from './genome';
import { InputNode } from './nodes/layers/input_node';
import { DenseNode } from './nodes/layers/dense_node';
import { OutputNode } from './nodes/layers/output_node';

describe('Genome', () => {
    it('initializes with input and output nodes', () => {
        const input = new InputNode([28, 28, 3]);
        const output = new OutputNode([28, 28, 3]);
        const genome = new Genome([input], [output]);

        expect(genome.inputNodes.length).toBe(1);
        expect(genome.outputNodes.length).toBe(1);
    });

    it('extracts a random subgenome', () => {
        const input = new InputNode([10]);
        const d1 = new DenseNode(16, "relu", true);
        const d2 = new DenseNode(16, "relu", true);
        const d3 = new DenseNode(16, "relu", true);
        const output = new OutputNode([16]);

        input.AddNext(d1);
        d1.AddNext(d2);
        d2.AddNext(d3);
        d3.AddNext(output);

        const genome = new Genome([input], [output]);
        const subgenome = genome.GetRandomSubgenome();

        expect(subgenome.length).toBeGreaterThan(0);
        // Ensure clones are physically different
        expect(subgenome[0]).not.toBe(d1);
    });

    it('breeds two functional genomes', () => {
        const input1 = new InputNode([10]);
        const d1 = new DenseNode(16, "relu", true);
        const output1 = new OutputNode([16]);
        input1.AddNext(d1);
        d1.AddNext(output1);
        const g1 = new Genome([input1], [output1]);

        const input2 = new InputNode([10]);
        const d2 = new DenseNode(16, "relu", true);
        const d3 = new DenseNode(16, "relu", true);
        const output2 = new OutputNode([16]);
        input2.AddNext(d2);
        d2.AddNext(d3);
        d3.AddNext(output2);
        const g2 = new Genome([input2], [output2]);

        const result = g1.Breed(g2);

        // Since extraction is random, we just check that a result is generated properly
        expect(result).toBeDefined();
        if (result) {
            expect(result.genome).toBeDefined();
            expect(result.isValid).toBeDefined();
        }
    });
});
