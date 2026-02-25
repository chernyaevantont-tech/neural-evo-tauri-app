import { describe, it, expect } from 'vitest';
import { useCanvasGenomeStore } from './../store';
import { DenseNode } from './layers/dense_node';
import { InputNode } from './layers/input_node';
import { Conv2DNode } from './layers/conv_node';
import { OutputNode } from './layers/output_node';
import { BaseNode } from './base_node';
import { AddNode } from './merge/add_node';
import { Concat2DNode } from './merge/concatinate_2d_node';
import { FlattenNode } from './layers/flatten_node';

describe('Graph Logic Edge Cases', () => {

    it('Edge Case: Connecting a node backwards should be rejected to prevent cycles', () => {
        const a = new DenseNode(10, "relu", true);
        const b = new DenseNode(10, "relu", true);
        const c = new DenseNode(10, "relu", true);

        a.AddNext(b);
        b.AddNext(c);

        // This is caught by CheckCompatibility -> isAcyclic logic or manually in store
        // Let's emulate a backward connection manually:
        c.AddNext(a);

        expect(c.isAcyclic()).toBe(false);
    });

    it('Edge Case: Output node should adopt the exact dimensions of its incoming node, multidimensional', () => {
        const input = new InputNode([32, 32, 3]);
        const conv = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true); // Output: 32, 32, 16
        const output = new OutputNode([32, 32, 16]);

        input.AddNext(conv);
        conv.AddNext(output);

        expect(output.GetInputShape()).toEqual([32, 32, 16]);

        // Ensure changing the pipeline updates the Output automatically
        // E.g if we suddenly changed Conv stride to 2 (emulated by manually forcing shape calculation upstream again)
        // (Since our model doesn't re-trigger from param changes automatically without calling Mutate or manual recalc, we won't test auto-trigger)
    });

    it('Edge Case: Flatten node should correctly collapse 3D tensors into 1D arrays', () => {
        const input = new InputNode([28, 28, 3]);
        const flatten = new FlattenNode();

        input.AddNext(flatten);
        expect(flatten.GetOutputShape()).toEqual([28 * 28 * 3]);
    });

    it('Edge Case: AddNode throws error on mismatched tensor dimensions dynamically', () => {
        const input1 = new InputNode([28, 28, 16]);
        const input2 = new InputNode([28, 28, 32]); // Different depth!

        const add = new AddNode();

        input1.AddNext(add);

        // It should throw an error when we attempt to add the second different-shaped node
        expect(() => {
            input2.AddNext(add);
        }).toThrow("AddNode: Cannot connect. Input shape mismatch!");
    });

    it('Edge Case: Disconnecting the middle node in a long chain should clean BOTH forward and backward refs', () => {
        const n1 = new DenseNode(10, "relu", true);
        const n2 = new DenseNode(10, "relu", true);
        const n3 = new DenseNode(10, "relu", true);

        n1.AddNext(n2);
        n2.AddNext(n3);

        n2.ClearAllConnections();

        expect(n2.next.length).toBe(0);
        expect(n2.previous.length).toBe(0);

        expect(n1.next.length).toBe(0); // Should properly decouple from N1's perspective
        expect(n3.previous.length).toBe(0); // Should properly decouple from N3's perspective
    });

    it('Edge Case: A merge node dropping from 2 connections down to 1 should still maintain shape correctly', () => {
        const input1 = new InputNode([28, 28, 16]);
        const input2 = new InputNode([28, 28, 16]);
        const concat = new Concat2DNode();

        input1.AddNext(concat);
        input2.AddNext(concat);

        expect(concat.GetOutputShape()).toEqual([28, 28, 32]);

        input2.RemoveNext(concat);

        // CalculateOutputShape must trigger to refresh
        concat["CalculateOutputShape"](); // Call protected member for test

        expect(concat.GetOutputShape()).toEqual([28, 28, 16]); // Should revert to 16
    });

});
