import { BaseNode, ResourceCriteria } from "../base_node";

export class OutputNode extends BaseNode {
    constructor(inputShape: Array<number>) {
        super();
        this.inputShape = [...inputShape];
    }

    protected CalculateOutputShape(): void {}

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                input_shape: this.inputShape
            }
        })
    }
    
    GetResources(dtype: number): ResourceCriteria {
        return {flash: 0, ram: 0, macs: 0}
    }

    protected Mutate(mutation_options: Map<string, number>): void {}

    CheckCompability(node: BaseNode): Boolean {
        return this.previous.length == 0 && node.GetOutputShape().every((val, index) => val == this.inputShape[index]);
    }

    protected AddPrev(node: BaseNode): void {
        this.previous.push(node)
    }

    public GetNodeType = (): string => "Output";

    public Clone = (): BaseNode  => new OutputNode(
        [...this.inputShape],
    );
}