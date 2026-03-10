import { BaseNode, ResourceCriteria } from "../base_node";

export class DropoutNode extends BaseNode {
    private probability: number;

    constructor(probability: number = 0.5) {
        super();
        this.probability = Math.max(0, Math.min(1, probability));
        // Dropout doesn't change dimensions, so we initialize with a placeholder
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
        // Dropout has no trainable parameters or significant MACs
        return { flash: 0, ram: 0, macs: 0 };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        const probMutation = mutation_options.get("dropout_prob") || -1;
        if (Math.random() <= probMutation) {
            // Randomly nudge the probability
            this.probability = Math.max(0, Math.min(1, this.probability + (Math.random() - 0.5) * 0.2));
        }
    }

    public GetExpectedInputDimensions(): number | "any" {
        return "any";
    }

    public GetOutputDimensions(): number | "any" {
        return "any";
    }

    public GetNodeType = (): string => "Dropout";

    protected _CloneImpl = (): BaseNode => new DropoutNode(this.probability);

    public GetIsMerging = (): boolean => false;

    // Helper for UI
    public GetProbability = (): number => this.probability;
    public SetProbability = (val: number) => {
        this.probability = Math.max(0, Math.min(1, val));
    }
}
