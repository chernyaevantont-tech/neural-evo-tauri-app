import { BaseNode, ResourceCriteria } from "../base_node";

export class InputNode extends BaseNode{
    
    constructor (d1: number, d2?: number, d3?: number) {
        super()
        if (!(Number.isInteger(d1) && (Number.isInteger(d2) || d2 == undefined) && (Number.isInteger(d3)  || d2 == undefined))) {
            throw new Error("Input shape values must be integers")
        }
        this.outputShape = [d1]
        if (d2) this.outputShape.push(d2);
        if (d3) this.outputShape.push(d3);
    }

    protected CalculateOutputShape(): void {}

    GetInfo(): String {
        return JSON.stringify({
            node: "Input",
            params: {
                output_shape: this.outputShape
            }
        });
    }

    GetResources(dtype: Number): ResourceCriteria {
        return {flash: 0, ram: 0, macs: 0}
    }
    protected Mutate(mutation_options: Map<string, number>): void {}

    CheckCompability(node: BaseNode): Boolean {
        throw new Error("Method not implemented.");
    }

}