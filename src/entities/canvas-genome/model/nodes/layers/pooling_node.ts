import { BaseNode, ResourceCriteria } from "../base_node";
import { KernelSize, PoolType } from "../types";
import { RandomizeInteger } from "../../../../../lib/random";

export class PoolingNode extends BaseNode {
    private static poolTypes: PoolType[] = ["max", "avg"]

    private poolType: PoolType
    private kernelSize: KernelSize
    private stride: number
    private padding: number

    constructor(
        poolType: PoolType,
        kernelSize: KernelSize,
        stride: number,
        padding: number
    ) {
        super();
        this.poolType = poolType;
        this.kernelSize = { ...kernelSize };
        this.stride = stride;
        this.padding = padding;
        this.inputShape = new Array(3);
    }

    protected CalculateOutputShape(): void {
        const hOut = Math.floor((this.inputShape[0] + 2 * this.padding - this.kernelSize.h) / this.stride + 1)
        const wOut = Math.floor((this.inputShape[1] + 2 * this.padding - this.kernelSize.w) / this.stride + 1)
        this.outputShape = [hOut, wOut, this.inputShape[2]]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                pool_type: this.poolType,
                kernel_size: this.kernelSize,
                stride: this.stride,
                padding: this.padding,
            }
        })
    }

    GetResources(dtype: number): ResourceCriteria {
        const ram = (this.inputShape[0] * this.inputShape[1] * this.inputShape[2] +
            this.outputShape[0] * this.outputShape[1] * this.outputShape[2]) * dtype;
        const macs = (this.outputShape[0] * this.outputShape[1] * this.outputShape[2] *
            (this.kernelSize.h * this.kernelSize.w - 1));

        return { flash: 0, ram: ram, macs: macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("pooling_pool_type") || -1)) {
            this.poolType = PoolingNode.poolTypes[RandomizeInteger(0, 1)];
        }

        if (Math.random() <= (mutation_options.get("pooling_kernel_size") || -1)) {
            const kernel_size = 1 * 2 * RandomizeInteger(0, 3);
            this.kernelSize = { h: kernel_size, w: kernel_size };
        }
    }

    CheckCompability(node: BaseNode): Boolean {
        return node.previous.length == 0 &&
            (node.GetInputShape().length == 3 || node.GetNodeType() == "Output" || node.GetIsMerging()) &&
            this.isAcyclic();
    }

    CheckCompabilityDisconnected(node: BaseNode): Boolean {
        return (node.GetInputShape().length == 3 || node.GetNodeType() == "Output" || node.GetIsMerging());
    }

    public GetNodeType = (): string => "Pooling";

    public Clone = (): BaseNode => new PoolingNode(
        this.poolType,
        { ...this.kernelSize },
        this.stride,
        this.padding,
    );

    public GetIsMerging = (): boolean => false;
}