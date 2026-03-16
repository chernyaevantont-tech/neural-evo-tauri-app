import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";
import type { ActivationFunction } from "../types";

export class LSTMNode extends BaseNode {
    private hiddenUnits: number;
    private gateActivation: ActivationFunction;
    private cellActivation: ActivationFunction;
    private hiddenActivation: ActivationFunction;
    private useBias: boolean;

    constructor(
        hiddenUnits: number,
        gateActivation: ActivationFunction = "sigmoid",
        cellActivation: ActivationFunction = "tanh",
        hiddenActivation: ActivationFunction = "tanh",
        useBias: boolean = true
    ) {
        super();

        if (!(Number.isInteger(hiddenUnits) && hiddenUnits > 0)) {
            throw Error("hiddenUnits must be a positive integer");
        }

        this.hiddenUnits = hiddenUnits;
        this.gateActivation = gateActivation;
        this.cellActivation = cellActivation;
        this.hiddenActivation = hiddenActivation;
        this.useBias = useBias;

        // LSTM expects 2D input: [sequence_length, input_features]
        this.inputShape = new Array<number>(2);
    }

    protected CalculateOutputShape(): void {
        // LSTM output shape: [sequence_length, hidden_units]
        // For each timestep, output is the hidden state
        this.outputShape = [this.inputShape[0], this.hiddenUnits];
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                hidden_units: this.hiddenUnits,
                gate_activation: this.gateActivation,
                cell_activation: this.cellActivation,
                hidden_activation: this.hiddenActivation,
                use_bias: this.useBias,
            },
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        const inputSize = this.inputShape[1];
        const hidden = this.hiddenUnits;

        // LSTM parameters per gate (4 gates: input, forget, cell, output):
        // Weight: (input_size + hidden_size) * hidden_size per gate
        // Bias: hidden_size per gate
        // Total: 4 * [(input_size + hidden_size) * hidden_size + hidden_size]
        const paramsPerGate =
            (inputSize + hidden) * hidden + (this.useBias ? hidden : 0);
        const flash = 4 * paramsPerGate * dtype;

        // Memory:
        // Input: seq_len * input_size
        // Output: seq_len * hidden_size
        // Hidden state: hidden_size
        // Cell state: hidden_size
        const ram =
            (this.inputShape[0] * inputSize +
                this.outputShape[0] * hidden +
                2 * hidden) *
            dtype;

        // Multiply-accumulate:
        // 4 gates * seq_len * (input_size + hidden_size) * hidden_size
        const macs =
            4 *
            this.inputShape[0] *
            (inputSize + hidden) *
            hidden;

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        // Mutate hidden units: [16, 512] in powers of 2
        if (Math.random() <= (mutation_options.get("lstm_hidden_units") || -1)) {
            this.hiddenUnits = Math.pow(2, RandomizeInteger(4, 9)); // 16-512
        }

        // Mutate gate activation
        if (Math.random() <= (mutation_options.get("lstm_gate_activation") || -1)) {
            const activations: ActivationFunction[] = ["sigmoid", "relu", "leaky_relu"];
            this.gateActivation =
                activations[RandomizeInteger(0, activations.length - 1)];
        }

        // Mutate cell activation
        if (Math.random() <= (mutation_options.get("lstm_cell_activation") || -1)) {
            const activations: ActivationFunction[] = ["tanh", "sigmoid"];
            this.cellActivation =
                activations[RandomizeInteger(0, activations.length - 1)];
        }

        // Mutate hidden activation
        if (Math.random() <= (mutation_options.get("lstm_hidden_activation") || -1)) {
            const activations: ActivationFunction[] = ["tanh", "sigmoid"];
            this.hiddenActivation =
                activations[RandomizeInteger(0, activations.length - 1)];
        }

        // Mutate bias: toggle
        if (Math.random() <= (mutation_options.get("lstm_use_bias") || -1)) {
            this.useBias = !this.useBias;
        }

        this.CalculateOutputShape();
    }

    public GetExpectedInputDimensions(): number | "any" {
        // LSTM expects 2D input: [sequence_length, input_features]
        return 2;
    }

    public GetOutputDimensions(): number | "any" {
        // LSTM outputs 2D: [sequence_length, hidden_units]
        return 2;
    }

    public GetNodeType = (): string => "LSTM";

    protected _CloneImpl = (): BaseNode =>
        new LSTMNode(
            this.hiddenUnits,
            this.gateActivation,
            this.cellActivation,
            this.hiddenActivation,
            this.useBias
        );

    public GetIsMerging = (): boolean => false;

    // Getters and setters for UI editing
    GetHiddenUnits(): number {
        return this.hiddenUnits;
    }

    SetHiddenUnits(units: number): void {
        if (units > 0) {
            this.hiddenUnits = units;
            this.CalculateOutputShape();
        }
    }

    GetGateActivation(): ActivationFunction {
        return this.gateActivation;
    }

    SetGateActivation(activation: ActivationFunction): void {
        this.gateActivation = activation;
    }

    GetCellActivation(): ActivationFunction {
        return this.cellActivation;
    }

    SetCellActivation(activation: ActivationFunction): void {
        this.cellActivation = activation;
    }

    GetHiddenActivation(): ActivationFunction {
        return this.hiddenActivation;
    }

    SetHiddenActivation(activation: ActivationFunction): void {
        this.hiddenActivation = activation;
    }

    GetUseBias(): boolean {
        return this.useBias;
    }

    SetUseBias(bias: boolean): void {
        this.useBias = bias;
    }
}
