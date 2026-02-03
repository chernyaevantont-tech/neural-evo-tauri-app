import { RandomizeInteger } from "../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";
import { ActivationFunction } from "../types";

export class DenseNode extends BaseNode {
    private static activationFunctions: ActivationFunction[] = ["relu", "leaky_relu", "softmax"]

    private units: number
    private activation: ActivationFunction
    private useBias: Boolean

    constructor(units: number, activation: ActivationFunction, useBias: Boolean) {
        super()
        if (!(Number.isInteger(units) && units >= 0)) {
            throw Error("units must be a positive number")
        }
        this.units = units
        this.activation = activation
        this.useBias = useBias
    }

    protected CalculateOutputShape(): void {
        this.outputShape = [this.units]
    }
    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                units: this.units,
                activation: this.activation,
                use_bias: this.useBias
            }
        });
    }
    GetResources(dtype: number): ResourceCriteria {
        const flash = this.outputShape[0] * (this.inputShape[0] + (this.useBias ? 1 : 0)) * dtype
        const ram = this.inputShape[0] * this.outputShape[0] * dtype
        const macs = this.inputShape[0] * this.outputShape[0]

        return { flash: flash, ram: ram, macs: macs }
    }
    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("dense_units") || -1)) {
            this.units = Math.pow(2, RandomizeInteger(4, 12))
        }

        this.CalculateOutputShape()

        if (Math.random() <= (mutation_options.get("dense_activation") || -1)) {
            this.activation = DenseNode.activationFunctions[RandomizeInteger(0, 2)]
        }

        if (Math.random() <= (mutation_options.get("dense_use_bias") || -1)) {
            this.useBias = !this.useBias
        }
    }
    CheckCompability(node: BaseNode): Boolean {
        return this.previous.length == 0 && node.GetOutputShape().length == 1
    }

    public GetNodeType = (): string => "Dense";
}