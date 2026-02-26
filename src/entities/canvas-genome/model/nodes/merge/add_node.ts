import { BaseNode, ResourceCriteria } from "../base_node";

export class AddNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.previous.length > 0) {
            this.outputShape = [...this.previous[0].GetOutputShape()]
        } else {
            this.outputShape = [...this.inputShape]
        }
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
        const checkShape = (this.outputShape && this.outputShape.length > 0) ? this.outputShape : this.inputShape;
        return (checkShape && checkShape.length > 0) ? checkShape.length : "any";
    }

    protected AddPrev(node: BaseNode): void {
        // Enforce that incoming shape matches the first connected node exactly
        if (this.previous.length > 0) {
            const firstShape = this.previous[0].GetOutputShape();
            const incShape = node.GetOutputShape();

            if (firstShape.length !== incShape.length || !firstShape.every((val, index) => val === incShape[index])) {
                throw new Error("AddNode: Cannot connect. Input shape mismatch!");
            }
        } else {
            this.inputShape = [...node.GetOutputShape()];
        }

        super.AddPrev(node);
    }

    public GetNodeType = (): string => "Add";

    protected _CloneImpl = (): BaseNode => new AddNode();

    public GetIsMerging = (): boolean => true;
}