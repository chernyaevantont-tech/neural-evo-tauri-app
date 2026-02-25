import { describe, it, expect } from 'vitest';
import { DenseNode } from './dense_node';
import { InputNode } from './input_node';
import { Conv2DNode } from './conv_node';

describe('DenseNode', () => {
    it('sets input shape and calc logic correctly', () => {
        const input = new InputNode([128]);
        const dense = new DenseNode(64, 'relu', true);

        input.AddNext(dense);

        expect(dense.GetInputShape()).toEqual([128]);
        expect(dense.GetOutputShape()).toEqual([64]);
    });

    it('only accepts 1D input or Output/Merge nodes', () => {
        const dense1D = new DenseNode(128, 'relu', true);
        const denseTarget = new DenseNode(64, 'relu', true);

        const conv3D = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);

        // the source node checks if the target node can accept it
        // fromNode.CheckCompability(toNode)
        expect(dense1D.CheckCompability(denseTarget)).toBe(true);
        expect(conv3D.CheckCompability(denseTarget)).toBe(false);
    });

    it('does not accept multiple inputs', () => {
        const input1 = new DenseNode(128, 'relu', true);
        const input2 = new DenseNode(128, 'relu', true);
        const dense = new DenseNode(64, 'relu', true);

        input1.AddNext(dense);

        // Since dense already has an input (input1), it should reject input2
        expect(input2.CheckCompability(dense)).toBe(false);
    });
});
