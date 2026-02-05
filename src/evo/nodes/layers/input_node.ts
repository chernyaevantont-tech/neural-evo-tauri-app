import { BaseNode, ResourceCriteria } from "../base_node";

export class InputNode extends BaseNode{
    
    constructor (outputShape: number[]) {
        super()
        this.outputShape = [...outputShape]
    }

    protected CalculateOutputShape(): void {}

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                output_shape: this.outputShape
            }
        });
    }

    GetResources(_dtype: Number): ResourceCriteria {
        return {flash: 0, ram: 0, macs: 0}
    }
    protected Mutate(_mutation_options: Map<string, number>): void {}

    CheckCompability(_node: BaseNode): Boolean {
        return false;
    }

    CheckCompabilityDisconnected(_node: BaseNode): Boolean {
        return false;
    }

    public GetNodeType = (): string => "Input";

    public Clone = (): BaseNode  => new InputNode(
        [...this.outputShape]
    );
}