import { BaseNode, ResourceCriteria } from "../base_node";

export class Concat2DNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.previous.length === 0) {
            this.inputShape = [];
            this.outputShape = [];
            return;
        }

        const firstShape = this.previous[0].GetOutputShape();
        if (firstShape.length < 3) {
            this.inputShape = [...firstShape];
            this.outputShape = [...firstShape];
            return;
        }

        const h = firstShape[0];
        const w = firstShape[1];

        // Sum channels across all inputs, guarding against invalid shapes
        const c = this.previous.reduce((sum, node) => {
            const s = node.GetOutputShape();
            return sum + (s.length === 3 ? s[2] : 0);
        }, 0);

        this.inputShape = [h, w, 0];
        this.outputShape = [h, w, c];
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

    public CanAcceptConnectionFrom(node: BaseNode, isDisconnectedCheck: boolean = false): boolean {
        if (!super.CanAcceptConnectionFrom(node, isDisconnectedCheck)) return false;

        const targetShape = this.GetInputShape();
        const incShape = node.GetOutputShape();

        // Concat2D concatenates on the channel axis (index 2)
        // So H and W (indices 0 and 1) must match exactly
        if (targetShape && targetShape.length === 3 && incShape && incShape.length === 3) {
            if (targetShape[0] !== incShape[0] || targetShape[1] !== incShape[1]) {
                return false;
            }
        }

        return true;
    }

    public GetNodeType = () => "Concat";

    protected _CloneImpl = (): BaseNode => new Concat2DNode();

    public GetIsMerging = (): boolean => true;
} 