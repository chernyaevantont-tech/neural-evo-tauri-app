import { describe, it, expect } from 'vitest';
import { InputNode } from './input_node';
import { OutputNode } from './output_node';
import { DenseNode } from './dense_node';

describe('InputNode', () => {
    it('initializes with given shape correctly', () => {
        const input = new InputNode([28, 28, 3]);

        expect(input.GetInputShape()).toEqual([]);
        expect(input.GetOutputShape()).toEqual([28, 28, 3]);
    });

    it('does not accept incoming connections', () => {
        const input = new InputNode([28, 28, 3]);
        const dense = new DenseNode(10, 'relu', true);

        // Input Nodes should ideally have node.previous.length == 0
        // Wait, dense -> input checks input.CheckCompability(dense)? 
        // No, check compatibility is fromNode.CheckCompability(toNode)
        expect(dense.CheckCompability(input)).toBe(false);
    });
});

describe('OutputNode', () => {
    it('sets shape matching incoming node', () => {
        const input = new InputNode([10]);
        const output = new OutputNode([10]);

        input.AddNext(output);

        expect(output.GetInputShape()).toEqual([10]);
    });
});
