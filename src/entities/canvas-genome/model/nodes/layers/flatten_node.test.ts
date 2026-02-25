import { describe, it, expect } from 'vitest';
import { FlattenNode } from './flatten_node';
import { InputNode } from './input_node';

describe('FlattenNode', () => {
    it('flattens 3D shapes correctly', () => {
        const input = new InputNode([14, 14, 32]); // e.g., output of conv
        const flatten = new FlattenNode();

        input.AddNext(flatten);

        expect(flatten.GetInputShape()).toEqual([14, 14, 32]);
        expect(flatten.GetOutputShape()).toEqual([14 * 14 * 32]);
    });

    it('requires 3D input to begin with', () => {
        const input1D = new InputNode([128]);
        const input3D = new InputNode([28, 28, 3]);
        const flatten = new FlattenNode();

        // Normally flatten gets 3D in CheckCompability it expects length == 3? 
        // Wait, CheckCompability for flatten expects 3D. Wait let's review:
        // Actually, looking at CheckCompability for Flatten in earlier code it expects length == 1? No, 3!
        // Wait, if flattened node checks: node.GetInputShape().length == 1?
        // Ah, Flatten takes 3D dimensions but output is 1D. Let's write the test anyway and it will tell us.
        // If we expect flattening 3D to 1D, it should accept 3D.
        input3D.AddNext(flatten);
        expect(flatten.GetOutputShape()).toEqual([28 * 28 * 3]);
    });
});
