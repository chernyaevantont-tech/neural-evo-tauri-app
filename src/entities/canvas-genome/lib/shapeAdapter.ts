import { BaseNode } from "../model/nodes/base_node";
import { Conv2DNode } from "../model/nodes/layers/conv_node";
import { DenseNode } from "../model/nodes/layers/dense_node";
import { FlattenNode } from "../model/nodes/layers/flatten_node";
import { PoolingNode } from "../model/nodes/layers/pooling_node";
import type { KernelSize } from "../model/nodes/types";

/**
 * ShapeAdapter Factory
 * 
 * Creates adapter layers to bridge incompatible tensor shapes during
 * evolutionary operations (mutations, crossover).
 * 
 * @see Genome.FindInsertionPoint - uses adapters for shape matching
 */
export class ShapeAdapterFactory {
    /**
     * Creates adapter layers to transform from one shape to another
     * @param fromShape - Source tensor shape
     * @param toShape - Target tensor shape
     * @returns Array of adapter layers, or null if transformation is not supported
     */
    static createAdapter(fromShape: number[], toShape: number[]): BaseNode[] | null {
        const adapters: BaseNode[] = [];

        // Case 1: 3D -> 3D (spatial dimensions or channels mismatch)
        if (fromShape.length === 3 && toShape.length === 3) {
            const [fromH, fromW, fromC] = fromShape;
            const [toH, toW, toC] = toShape;

            // Handle spatial dimension mismatch
            if (fromH !== toH || fromW !== toW) {
                const needDownsample = fromH > toH || fromW > toW;

                if (needDownsample) {
                    const hRatio = fromH / toH;
                    const wRatio = fromW / toW;
                    const stride = Math.max(2, Math.floor(Math.min(hRatio, wRatio)));
                    const kernelSize: KernelSize = { h: stride, w: stride };

                    adapters.push(new PoolingNode('max', kernelSize, stride, 0));

                    const newH = Math.floor((fromH - kernelSize.h) / stride + 1);
                    const newW = Math.floor((fromW - kernelSize.w) / stride + 1);

                    if (newH !== toH || newW !== toW) {
                        const remainingHRatio = newH / toH;
                        const remainingWRatio = newW / toW;
                        const convStride = Math.max(1, Math.floor(Math.min(remainingHRatio, remainingWRatio)));

                        adapters.push(new Conv2DNode(
                            fromC,
                            { h: 3, w: 3 },
                            convStride,
                            1,
                            1,
                            true,
                            'relu'
                        ));
                    }
                } else {
                    return null; // Upsampling not supported
                }
            }

            // Handle channel mismatch
            if (fromC !== toC) {
                adapters.push(new Conv2DNode(
                    toC,
                    { h: 1, w: 1 },
                    1,
                    0,
                    1,
                    true,
                    'relu'
                ));
            }

            return adapters.length > 0 ? adapters : null;
        }

        // Case 2: 3D -> 1D (Flatten + optional Dense)
        if (fromShape.length === 3 && toShape.length === 1) {
            adapters.push(new FlattenNode());

            const flattenedSize = fromShape[0] * fromShape[1] * fromShape[2];
            if (flattenedSize !== toShape[0]) {
                adapters.push(new DenseNode(toShape[0], 'relu', true));
            }

            return adapters;
        }

        // Case 3: 1D -> 1D (dimension mismatch)
        if (fromShape.length === 1 && toShape.length === 1) {
            if (fromShape[0] !== toShape[0]) {
                adapters.push(new DenseNode(toShape[0], 'relu', false));
                return adapters;
            }
            return null;
        }

        // Case 4: 1D -> 3D (not supported)
        if (fromShape.length === 1 && toShape.length === 3) {
            return null;
        }

        // Unknown case
        return null;
    }
}
