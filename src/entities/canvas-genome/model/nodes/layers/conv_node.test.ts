import { describe, it, expect } from 'vitest';
import { Conv2DNode } from './conv_node';
import { InputNode } from './input_node';
import { DenseNode } from './dense_node';
import { PoolingNode } from './pooling_node';

describe('Conv2DNode', () => {
    it('sets input shape and calc logic correctly', () => {
        // e.g. 28x28x3 input
        const input = new InputNode([28, 28, 3]);
        // 16 filters, 3x3 kernel, 1 stride, 1 padding, 1 dilation, true bias
        const conv = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);

        input.AddNext(conv);

        expect(conv.GetInputShape()).toEqual([28, 28, 3]);
        // out_h = (28 + 2*1 - 1*(3-1) - 1)/1 + 1 = 28
        // out_w = (28 + 2*1 - 1*(3-1) - 1)/1 + 1 = 28
        expect(conv.GetOutputShape()).toEqual([28, 28, 16]);
    });

    it('calculates complex shapes correctly', () => {
        const input = new InputNode([32, 32, 3]);
        // 32 filters, 5x5 kernel, 2 stride, 0 padding, 1 dilation
        const conv = new Conv2DNode(32, { h: 5, w: 5 }, 2, 0, 1, true);

        input.AddNext(conv);
        // out = math.floor((32 + 0 - 1*(5-1) - 1)/2 + 1) -> (32 - 4 - 1)/2 + 1 = 27/2+1 = 13 + 1 = 14
        expect(conv.GetOutputShape()).toEqual([14, 14, 32]);
    });

    it('only accepts 3D input', () => {
        const dense = new DenseNode(128, 'relu', true);
        const pool = new PoolingNode("max", { h: 2, w: 2 }, 2, 0);
        const conv = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);

        expect(dense.CheckCompability(conv)).toBe(false);
        expect(pool.CheckCompability(conv)).toBe(true);
    });

    it('does not accept multiple inputs', () => {
        const convSrc1 = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);
        const convSrc2 = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);
        const conv = new Conv2DNode(16, { h: 3, w: 3 }, 1, 1, 1, true);

        convSrc1.AddNext(conv);
        expect(convSrc2.CheckCompability(conv)).toBe(false);
    });
});
