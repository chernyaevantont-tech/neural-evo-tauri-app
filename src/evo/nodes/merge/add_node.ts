import { BaseNode, ResourceCriteria } from "../base_node";

export class AddNode extends BaseNode {
    protected CalculateOutputShape(): void {
        this.outputShape = this.inputShape
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {}
        })
    }

    GetResources(_dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }
    }

    protected Mutate(_mutation_options: Map<string, number>): void { }

    CheckCompability(node: BaseNode): Boolean {
        return this.inputShape.length == 0 ? true :
            node.GetOutputShape().every((val, index) => val == this.inputShape[index])
            && this.isAcyclic();
    }

    CheckCompabilityDisconnected(node: BaseNode): Boolean {
        return this.previous.length == 1 ? true : node.GetOutputShape().every((val, index) => val == this.inputShape[index])
    }

    public GetNodeType = (): string => "Add";

    public Clone = (): BaseNode => new AddNode();
}