import { BaseNode, ResourceCriteria } from "../base_node";

export class BatchNormNode extends BaseNode {
    private epsilon: number;
    private momentum: number;

    constructor(epsilon: number = 1e-5, momentum: number = 0.1) {
        super();
        this.epsilon = epsilon;
        this.momentum = momentum;
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
                epsilon: this.epsilon,
                momentum: this.momentum
            }
        });
    }

    GetResources(_dtype: number): ResourceCriteria {
        // BatchNorm has trainable parameters (gamma, beta) proportional to channels
        const channels = this.inputShape.length > 0 ? this.inputShape[0] : 0;
        return {
            flash: channels * 2 * 4, // gamma, beta (float32)
            ram: channels * 4 * 4,   // gamma, beta, running_mean, running_var
            macs: channels // approximate
        };
    }

    protected Mutate(_mutation_options: Map<string, number>): void {
        // Mutations for BatchNorm parameters could be added here if needed
    }

    public GetExpectedInputDimensions(): number | "any" {
        return "any"; // Supports 1D (Dense) or 3D (Conv)
    }

    public GetOutputDimensions(): number | "any" {
        return (this.outputShape && this.outputShape.length > 0) ? this.outputShape.length : "any";
    }

    public GetNodeType = (): string => "BatchNorm";

    protected _CloneImpl = (): BaseNode => new BatchNormNode(this.epsilon, this.momentum);

    public GetIsMerging = (): boolean => false;

    // Accessors
    public GetEpsilon = () => this.epsilon;
    public SetEpsilon = (val: number) => this.epsilon = val;
    public GetMomentum = () => this.momentum;
    public SetMomentum = (val: number) => this.momentum = val;
}
