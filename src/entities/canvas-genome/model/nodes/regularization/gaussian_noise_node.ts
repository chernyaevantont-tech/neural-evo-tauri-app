import { BaseNode, ResourceCriteria } from "../base_node";

export class GaussianNoiseNode extends BaseNode {
    private stdDev: number;

    constructor(stdDev: number = 0.1) {
        super();
        this.stdDev = Math.max(0, stdDev);
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
                std_dev: this.stdDev
            }
        });
    }

    GetResources(_dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 };
    }

    protected Mutate(_mutation_options: Map<string, number>): void {
        // Gaussian noise std_dev mutation logic can be added here
    }

    public GetExpectedInputDimensions(): number | "any" {
        return "any";
    }

    public GetOutputDimensions(): number | "any" {
        return (this.outputShape && this.outputShape.length > 0) ? this.outputShape.length : "any";
    }

    public GetNodeType = (): string => "GaussianNoise";

    protected _CloneImpl = (): BaseNode => new GaussianNoiseNode(this.stdDev);

    public GetIsMerging = (): boolean => false;

    // Accessors
    public GetStdDev = () => this.stdDev;
    public SetStdDev = (val: number) => this.stdDev = Math.max(0, val);
}
