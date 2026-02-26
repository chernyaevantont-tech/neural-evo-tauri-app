import { BaseNode, ResourceCriteria } from "../base_node";

export class Concat2DNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.previous.length == 0) {
            return
        }

        const h = this.previous[0].GetOutputShape()[0]
        const w = this.previous[0].GetOutputShape()[1]
        const c = this.previous.reduce((result, current) => result + current.GetOutputShape()[2], 0)

        this.inputShape = [h, w, 0];
        this.outputShape = [h, w, c]
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
        return 3;
    }

    public GetOutputDimensions(): number | "any" {
        return 3;
    }



    public GetNodeType = () => "Concat";

    protected _CloneImpl = (): BaseNode => new Concat2DNode();

    public GetIsMerging = (): boolean => true;
} 