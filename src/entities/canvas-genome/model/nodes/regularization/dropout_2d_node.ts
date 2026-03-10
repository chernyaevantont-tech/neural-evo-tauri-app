import { BaseNode, ResourceCriteria } from "../base_node";

/**
 * Spatial Dropout (Dropout2D) - drops entire channels.
 * Strictly requires 3D output (Spatial) from predecessor.
 */
export class Dropout2DNode extends BaseNode {
    private probability: number;

    constructor(probability: number = 0.5) {
        super();
        this.probability = Math.max(0, Math.min(1, probability));
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
                prob: this.probability
            }
        });
    }

    GetResources(_dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        const probMutation = mutation_options.get("dropout_prob") || -1;
        if (Math.random() <= probMutation) {
            this.probability = Math.max(0, Math.min(1, this.probability + (Math.random() - 0.5) * 0.2));
        }
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 3; // Strictly requires [C, H, W]
    }

    public GetOutputDimensions(): number | "any" {
        return 3;
    }

    public GetNodeType = (): string => "Dropout2D";

    protected _CloneImpl = (): BaseNode => new Dropout2DNode(this.probability);

    public GetIsMerging = (): boolean => false;

    // Accessors
    public GetProbability = (): number => this.probability;
    public SetProbability = (val: number) => {
        this.probability = Math.max(0, Math.min(1, val));
    }
}
