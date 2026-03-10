import { BaseNode, ResourceCriteria } from "../base_node";

export class LayerNormNode extends BaseNode {
    private epsilon: number;

    constructor(epsilon: number = 1e-5) {
        super();
        this.epsilon = epsilon;
        this.inputShape = [];
    }

    protected CalculateOutputShape(): void {
        if (this.previous.length > 0) {
            const firstInputShape = this.previous[0].GetOutputShape();
            this.inputShape = [...firstInputShape];
            this.outputShape = [...firstInputShape];
        } else {
            this.outputShape = [...this.inputShape];
        }
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                epsilon: this.epsilon
            }
        });
    }

    GetResources(_dtype: number): ResourceCriteria {
        const features = this.inputShape.length > 0 ? this.inputShape.reduce((a, b) => a * b, 1) : 0;
        return {
            flash: features * 2 * 4,
            ram: features * 2 * 4,
            macs: features
        };
    }

    protected Mutate(_mutation_options: Map<string, number>): void { }

    public GetExpectedInputDimensions(): number | "any" {
        return "any";
    }

    public GetOutputDimensions(): number | "any" {
        return (this.outputShape && this.outputShape.length > 0) ? this.outputShape.length : "any";
    }

    public GetNodeType = (): string => "LayerNorm";

    protected _CloneImpl = (): BaseNode => new LayerNormNode(this.epsilon);

    public GetIsMerging = (): boolean => false;

    // Accessors
    public GetEpsilon = () => this.epsilon;
    public SetEpsilon = (val: number) => this.epsilon = val;
}
