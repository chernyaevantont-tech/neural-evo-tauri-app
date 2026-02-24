import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";
import { KernelSize } from "../types";

export class Conv2DNode extends BaseNode {
    private static dilationOptions = [1, 2, 4, 8]

    private filters: number
    private kernelSize: KernelSize
    private stride: number
    private padding: number
    private dilation: number
    private useBias: boolean

    constructor(
        filters: number,
        kernelSize: KernelSize,
        stride: number,
        padding: number,
        dilation: number,
        useBias: boolean
    ) {
        super()

        this.filters = filters
        this.kernelSize = kernelSize
        this.stride = stride
        this.padding = padding
        this.dilation = dilation
        this.useBias = useBias
    }

    protected CalculateOutputShape(): void {
        const hOut = Math.floor((this.inputShape[0] + 2 * this.padding - this.dilation * (this.kernelSize.h - 1) - 1) / this.stride + 1)
        const wOut = Math.floor((this.inputShape[1] + 2 * this.padding - this.dilation * (this.kernelSize.w - 1) - 1) / this.stride + 1)
        this.outputShape = [hOut, wOut, this.filters]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                filters: this.filters,
                kernel_size: this.kernelSize,
                stride: this.stride,
                padding: this.padding,
                dilation: this.dilation,
                use_bias: this.useBias
            }
        })
    }

    GetResources(dtype: number): ResourceCriteria {
        const flash = this.outputShape[2] * (this.kernelSize.h * this.kernelSize.w * this.inputShape[2]
            + (this.useBias ? 1 : 0)) * dtype
        const ram = (this.inputShape[0] * this.inputShape[1] * this.inputShape[2]
            + this.outputShape[0] * this.outputShape[1] * this.outputShape[2]) * dtype
        const macs = (this.outputShape[0] * this.outputShape[1] * this.outputShape[2])
            * this.kernelSize.h * this.kernelSize.w * this.inputShape[2]

        return { flash: flash, ram: ram, macs: macs }
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("conv2d_filters") || -1)) {
            this.filters = 4 * RandomizeInteger(4, 16)
        }

        if (Math.random() <= (mutation_options.get("conv2d_kernel_size") || -1)) {
            const kernelSize = 1 + 2 * RandomizeInteger(0, 3)
            this.kernelSize = { h: kernelSize, w: kernelSize }
        }

        if (Math.random() <= (mutation_options.get("conv2d_stride_size") || -1)) {
            this.stride = RandomizeInteger(1, 2)
        }

        if (Math.random() <= (mutation_options.get("conv2d_padding") || -1)) {
            this.padding = RandomizeInteger(1, 2)
        }

        if (Math.random() <= (mutation_options.get("conv2d_dilation") || -1)) {
            this.dilation = Conv2DNode.dilationOptions[RandomizeInteger(0, Conv2DNode.dilationOptions.length - 1)]
        }

        if (Math.random() <= (mutation_options.get("conv2d_use_bias") || -1)) {
            this.useBias = !this.useBias
        }

        this.CalculateOutputShape()
    }

    CheckCompability(node: BaseNode): Boolean {
        return (node.previous.length == 0 || node.GetIsMerging()) &&
            node.GetInputShape().length == 3 &&
            this.isAcyclic();
    }

    CheckCompabilityDisconnected(node: BaseNode): Boolean {
        return (node.previous.length == 1 || node.GetIsMerging()) &&
            node.GetInputShape().length == 3 &&
            this.isAcyclic();
    }

    public GetNodeType = (): string => "Conv2D";

    public Clone = (): BaseNode => new Conv2DNode(
        this.filters,
        this.kernelSize,
        this.stride,
        this.padding,
        this.dilation,
        this.useBias
    );

    public GetIsMerging = (): boolean => false;
}