import { BaseNode, ResourceCriteria } from "../base_node";

export class OutputNode extends BaseNode {
    constructor(inputShape: Array<number>) {
        super();
        this.inputShape = [...inputShape];
    }

    protected CalculateOutputShape(): void { }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                input_shape: this.inputShape
            }
        })
    }

    GetResources(_dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }
    }

    protected Mutate(_mutation_options: Map<string, number>): void { }

    public GetExpectedInputDimensions(): number | "any" {
        return this.inputShape.length;
    }

    public GetOutputDimensions(): number | "any" {
        return "any";
    }


    public GetNodeType = (): string => "Output";

    protected _CloneImpl = (): BaseNode => new OutputNode(
        [...this.inputShape],
    );

    public GetIsMerging = (): boolean => false;
}