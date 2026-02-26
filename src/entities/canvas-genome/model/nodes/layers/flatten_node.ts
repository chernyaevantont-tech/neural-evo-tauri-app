import { BaseNode, ResourceCriteria } from "../base_node";

export class FlattenNode extends BaseNode {
    constructor() {
        super();
        this.inputShape = new Array<number>(3);
    }

    protected CalculateOutputShape(): void {
        this.outputShape = [this.inputShape[0] * this.inputShape[1] * this.inputShape[2]]
    }
    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {}
        })
    }
    GetResources(dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }
    }
    protected Mutate(mutation_options: Map<string, number>): void { }

    public GetExpectedInputDimensions(): number | "any" {
        return 3;
    }

    public GetOutputDimensions(): number | "any" {
        return 1;
    }

    public GetNodeType = (): string => "Flatten";

    protected _CloneImpl = (): BaseNode => new FlattenNode();

    public GetIsMerging = (): boolean => false;
}