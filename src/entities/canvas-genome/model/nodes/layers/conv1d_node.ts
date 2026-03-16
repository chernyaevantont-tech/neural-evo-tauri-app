import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";

export class Conv1DNode extends BaseNode {
    private static dilationOptions = [1, 2, 4, 8];

    private filters: number;
    private kernelSize: number;
    private stride: number;
    private padding: number;
    private dilation: number;
    private useBias: boolean;
    private activation: string;

    constructor(
        filters: number,
        kernelSize: number,
        stride: number,
        padding: number,
        dilation: number,
        useBias: boolean,
        activation: string = "relu"
    ) {
        super();

        if (!(Number.isInteger(filters) && filters > 0)) {
            throw Error("filters must be a positive integer");
        }
        if (!(Number.isInteger(kernelSize) && kernelSize > 0)) {
            throw Error("kernelSize must be a positive integer");
        }
        if (!(Number.isInteger(stride) && stride > 0)) {
            throw Error("stride must be a positive integer");
        }
        if (!(Number.isInteger(padding) && padding >= 0)) {
            throw Error("padding must be non-negative");
        }
        if (!(Number.isInteger(dilation) && dilation > 0)) {
            throw Error("dilation must be a positive integer");
        }

        this.filters = filters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.useBias = useBias;
        this.activation = activation;
        
        // Conv1D expects 2D input: [sequence_length, input_channels]
        this.inputShape = new Array<number>(2);
    }

    protected CalculateOutputShape(): void {
        // Output sequence length formula:
        // L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        const seqLenOut = Math.floor(
            (this.inputShape[0] + 2 * this.padding - this.dilation * (this.kernelSize - 1) - 1) / 
            this.stride + 1
        );
        
        // Output shape: [sequence_length_out, filters]
        this.outputShape = [seqLenOut, this.filters];
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
                use_bias: this.useBias,
                activation: this.activation,
            },
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        // Model parameters (flash):
        // Weight: filters * kernel_size * input_channels + bias
        const flash =
            this.filters *
            (this.kernelSize * this.inputShape[1] + (this.useBias ? 1 : 0)) *
            dtype;

        // Memory (RAM):
        // Input: seq_len * input_channels
        // Output: seq_len_out * filters
        const ram =
            (this.inputShape[0] * this.inputShape[1] +
                this.outputShape[0] * this.outputShape[1]) *
            dtype;

        // Multiply-accumulate operations:
        // output_seq_len * filters * kernel_size * input_channels
        const macs =
            this.outputShape[0] *
            this.filters *
            this.kernelSize *
            this.inputShape[1];

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        // Mutate filters: typically powers of 2 (16, 32, 64, 128, ...)
        if (Math.random() <= (mutation_options.get("conv1d_filters") || -1)) {
            this.filters = Math.pow(2, RandomizeInteger(4, 7)); // 16-128
        }

        // Mutate kernel size: 3, 5, 7, 9, 11
        if (Math.random() <= (mutation_options.get("conv1d_kernel_size") || -1)) {
            this.kernelSize = 1 + 2 * RandomizeInteger(1, 5); // 3, 5, 7, 9, 11
        }

        // Mutate stride: 1, 2
        if (Math.random() <= (mutation_options.get("conv1d_stride") || -1)) {
            this.stride = RandomizeInteger(1, 2); // 1 or 2
        }

        // Mutate padding: 0, 1, 2
        if (Math.random() <= (mutation_options.get("conv1d_padding") || -1)) {
            this.padding = RandomizeInteger(0, 2);
        }

        // Mutate dilation: choose from [1, 2, 4, 8]
        if (Math.random() <= (mutation_options.get("conv1d_dilation") || -1)) {
            this.dilation = Conv1DNode.dilationOptions[
                RandomizeInteger(0, Conv1DNode.dilationOptions.length - 1)
            ];
        }

        // Mutate bias: toggle
        if (Math.random() <= (mutation_options.get("conv1d_use_bias") || -1)) {
            this.useBias = !this.useBias;
        }

        // Mutate activation: choose from list
        if (Math.random() <= (mutation_options.get("conv1d_activation") || -1)) {
            const activations = ["relu", "leaky_relu", "sigmoid", "tanh", "linear"];
            this.activation = activations[RandomizeInteger(0, activations.length - 1)];
        }

        this.CalculateOutputShape();
    }

    public GetExpectedInputDimensions(): number | "any" {
        // Conv1D expects 2D input: [sequence_length, input_channels]
        return 2;
    }

    public GetOutputDimensions(): number | "any" {
        // Conv1D outputs 2D: [sequence_length_out, filters]
        return 2;
    }

    public GetNodeType = (): string => "Conv1D";

    protected _CloneImpl = (): BaseNode =>
        new Conv1DNode(
            this.filters,
            this.kernelSize,
            this.stride,
            this.padding,
            this.dilation,
            this.useBias,
            this.activation
        );

    public GetIsMerging = (): boolean => false;

    // Getters for UI editing
    GetFilters(): number {
        return this.filters;
    }

    SetFilters(filters: number): void {
        if (filters > 0) {
            this.filters = filters;
            this.CalculateOutputShape();
        }
    }

    GetKernelSize(): number {
        return this.kernelSize;
    }

    SetKernelSize(kernelSize: number): void {
        if (kernelSize > 0) {
            this.kernelSize = kernelSize;
            this.CalculateOutputShape();
        }
    }

    GetStride(): number {
        return this.stride;
    }

    SetStride(stride: number): void {
        if (stride > 0) {
            this.stride = stride;
            this.CalculateOutputShape();
        }
    }

    GetPadding(): number {
        return this.padding;
    }

    SetPadding(padding: number): void {
        if (padding >= 0) {
            this.padding = padding;
            this.CalculateOutputShape();
        }
    }

    GetDilation(): number {
        return this.dilation;
    }

    SetDilation(dilation: number): void {
        if (dilation > 0) {
            this.dilation = dilation;
            this.CalculateOutputShape();
        }
    }

    GetUseBias(): boolean {
        return this.useBias;
    }

    SetUseBias(useBias: boolean): void {
        this.useBias = useBias;
    }

    GetActivation(): string {
        return this.activation;
    }

    SetActivation(activation: string): void {
        this.activation = activation;
    }
}
