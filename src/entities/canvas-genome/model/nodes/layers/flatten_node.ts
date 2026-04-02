import { BaseNode, ResourceCriteria } from "../base_node";

export class FlattenNode extends BaseNode {
    constructor() {
        super();
        this.inputShape = [];
    }

    protected CalculateOutputShape(): void {
        const total = this.inputShape.reduce((a, b) => a * b, 1);
        this.outputShape = [total];
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

    public GetExpectedInputDimensions(): number | "any" {
        return "any";
    }

    public GetOutputDimensions(): number | "any" {
        return 1;
    }

    public GetNodeType = (): string => "Flatten";

    protected _CloneImpl = (): BaseNode => new FlattenNode();

    public GetIsMerging = (): boolean => false;
}