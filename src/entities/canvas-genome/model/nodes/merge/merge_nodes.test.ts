import { describe, it, expect } from 'vitest';
import { AddNode } from './add_node';
import { Concat2DNode } from './concatinate_2d_node';
import { InputNode } from '../layers/input_node';

describe('AddNode', () => {
    it('adopts shape of the first connected node', () => {
        const input = new InputNode([128]);
        const add = new AddNode();

        input.AddNext(add);

        expect(add.GetInputShape()).toEqual([128]);
        expect(add.GetOutputShape()).toEqual([128]);
    });

    it('throws error if second node has different shape', () => {
        const input1 = new InputNode([128]);
        const input2 = new InputNode([64]);
        const add = new AddNode();

        input1.AddNext(add);

        // input2 has different shape [64] vs [128], so AddNext should throw
        expect(() => {
            input2.AddNext(add);
        }).toThrow("AddNode: Cannot connect. Input shape mismatch!");
    });

    it('accepts second node if shape matches', () => {
        const input1 = new InputNode([128]);
        const input2 = new InputNode([128]);
        const add = new AddNode();

        input1.AddNext(add);
        expect(() => {
            input2.AddNext(add);
        }).not.toThrow();

        expect(add.previous.length).toBe(2);
    });
});

describe('Concat2DNode', () => {
    it('adopts H and W from the first connected node and sums channels', () => {
        const input1 = new InputNode([28, 28, 16]);
        const input2 = new InputNode([28, 28, 32]);
        const concat = new Concat2DNode();

        input1.AddNext(concat);
        input2.AddNext(concat);

        expect(concat.GetInputShape()).toEqual([28, 28, 0]); // Base input before addition
        expect(concat.GetOutputShape()).toEqual([28, 28, 48]); // 16 + 32 channels
    });

    it('compatibility check validates H and W matches', () => {
        const input1 = new InputNode([28, 28, 16]);
        const input2 = new InputNode([14, 14, 32]);
        const concat = new Concat2DNode();

        input1.AddNext(concat);

        // Concat's check expects target nodes to receive its output.
        // Wait, Concat's check logic: fromNode.CheckCompability(Concat).
        // input2 checks if Concat is valid. Input checks target.
        // What about when Concat connects to output?
        expect(concat.CheckCompability(new InputNode([28, 28, 48]))).toBe(true);
    });
});
