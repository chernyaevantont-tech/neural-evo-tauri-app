import { describe, it, expect } from 'vitest';
import { PoolingNode } from './pooling_node';
import { InputNode } from './input_node';
import { DenseNode } from './dense_node';

describe('PoolingNode', () => {
    it('calculates output shape for Max pool correctly', () => {
        const input = new InputNode([28, 28, 16]);
        // pool max, 2x2 kernel, 2 stride, 0 padding
        const pool = new PoolingNode("max", { h: 2, w: 2 }, 2, 0);

        input.AddNext(pool);

        expect(pool.GetInputShape()).toEqual([28, 28, 16]);
        // out_h = floor((28 + 0 - 2)/2 + 1) = 14
        expect(pool.GetOutputShape()).toEqual([14, 14, 16]);
    });

    it('only accepts 3D input', () => {
        const dense = new DenseNode(128, 'relu', true);
        const poolSrc = new PoolingNode("avg", { h: 2, w: 2 }, 2, 0);
        const pool = new PoolingNode("avg", { h: 2, w: 2 }, 2, 0);

        expect(dense.CheckCompability(pool)).toBe(false);
        expect(poolSrc.CheckCompability(pool)).toBe(true);
    });
});
