import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";
import type { ActivationFunction } from "../types";

export class MultiHeadAttentionNode extends BaseNode {
    private nHeads: number;
    private dropout: number;
    private quietSoftmax: boolean;

    constructor(
        nHeads: number = 4,
        dropout: number = 0.1,
        quietSoftmax: boolean = false
    ) {
        super();

        if (!(Number.isInteger(nHeads) && nHeads > 0)) {
            throw Error("nHeads must be a positive integer");
        }
        if (!(dropout >= 0 && dropout <= 1)) {
            throw Error("dropout must be in [0, 1]");
        }

        this.nHeads = nHeads;
        this.dropout = dropout;
        this.quietSoftmax = quietSoftmax;

        // Multi-head attention expects 2D input: [sequence_length, d_model]
        this.inputShape = new Array<number>(2);
    }

    protected CalculateOutputShape(): void {
        // MHA output shape equals input shape: [sequence_length, d_model]
        this.outputShape = [...this.inputShape];
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                n_heads: this.nHeads,
                dropout: this.dropout,
                quiet_softmax: this.quietSoftmax,
            },
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        const [seqLen, dModel] = this.inputShape;

        // Multi-head attention parameters:
        // 4 projections (query, key, value, output): each is Linear(d_model, d_model)
        // Weight: d_model * d_model per projection
        // Bias: d_model per projection
        const flash = 4 * (dModel * dModel + dModel) * dtype;

        // Memory:
        // Input: seq_len * d_model
        // Output: seq_len * d_model
        // Attention scores: seq_len * seq_len (temporary)
        const ram = (2 * seqLen * dModel + seqLen * seqLen) * dtype;

        // Multiply-accumulate:
        // Query, Key, Value projections: 3 * seq_len * d_model * d_model
        // Attention: seq_len * seq_len * d_model
        // Output projection: seq_len * d_model * d_model
        const macs =
            4 * seqLen * dModel * dModel + // 4 projections
            seqLen * seqLen * dModel; // attention + aggregation

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        const dModel = this.inputShape[1];
        const maxHeads = Math.min(Math.floor(dModel / 2), 16);

        // Mutate n_heads: [1, min(d_model/2, 16)]
        if (Math.random() <= (mutation_options.get("mha_n_heads") || -1)) {
            this.nHeads = Math.max(1, Math.floor(Math.random() * maxHeads));
        }

        // Mutate dropout: [0.0, 0.5]
        if (Math.random() <= (mutation_options.get("mha_dropout") || -1)) {
            this.dropout = Math.random() * 0.5;
        }

        // Mutate quiet_softmax: toggle with 20% probability
        if (Math.random() <= (mutation_options.get("mha_quiet_softmax") || -1)) {
            this.quietSoftmax = !this.quietSoftmax;
        }

        this.CalculateOutputShape();
    }

    public GetExpectedInputDimensions(): number | "any" {
        // MHA expects 2D input: [sequence_length, d_model]
        return 2;
    }

    public GetOutputDimensions(): number | "any" {
        // MHA outputs 2D: [sequence_length, d_model]
        return 2;
    }

    public GetNodeType = (): string => "MultiHeadAttention";

    protected _CloneImpl = (): BaseNode =>
        new MultiHeadAttentionNode(
            this.nHeads,
            this.dropout,
            this.quietSoftmax
        );

    public GetIsMerging = (): boolean => false;

    // Getters and setters for UI editing
    GetNHeads(): number {
        return this.nHeads;
    }

    SetNHeads(nHeads: number): void {
        if (nHeads > 0) {
            this.nHeads = nHeads;
            this.CalculateOutputShape();
        }
    }

    GetDropout(): number {
        return this.dropout;
    }

    SetDropout(dropout: number): void {
        if (dropout >= 0 && dropout <= 1) {
            this.dropout = dropout;
        }
    }

    GetQuietSoftmax(): boolean {
        return this.quietSoftmax;
    }

    SetQuietSoftmax(quietSoftmax: boolean): void {
        this.quietSoftmax = quietSoftmax;
    }
}
