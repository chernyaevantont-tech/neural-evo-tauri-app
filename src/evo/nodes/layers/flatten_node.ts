import { BaseNode, ResourceCriteria } from "../base_node";

export class FlattenNode extends BaseNode {
    protected CalculateOutputShape(): void {
        this.outputShape = [this.inputShape[0] * this.inputShape[1] * this.inputShape[2]]
    }
    GetInfo(): String {
        return JSON.stringify({
            node: "Flatten",
            params: {
                output_shape : this.outputShape
            }
        })
    }
    GetResources(dtype: number): ResourceCriteria {
       return {flash: 0, ram: 0, macs: 0}
    }
    protected Mutate(mutation_options: Map<string, number>): void {}

    CheckCompability(node: BaseNode): Boolean {
        return node.GetOutputShape().length == 3
    }

}