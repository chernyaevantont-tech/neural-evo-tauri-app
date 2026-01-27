import { BaseNode, ResourceCriteria } from "../base_node";

export class AddNode extends BaseNode {
    protected CalculateOutputShape(): void {
        this.outputShape = this.inputShape
    }

    GetInfo(): String {
        return JSON.stringify({
            node: "Add",
            params: {
                input_shape: this.inputShape,
                output_shape: this.outputShape
            }
        })
    }

    GetResources(dtype: number): ResourceCriteria {
        return {flash: 0, ram: 0, macs: 0}
    }

    protected Mutate(mutation_options: Map<string, number>): void {}
    
    CheckCompability(node: BaseNode): Boolean {
        return this.inputShape.length == 0 ? true: this.inputShape == node.GetOutputShape()
    }

}