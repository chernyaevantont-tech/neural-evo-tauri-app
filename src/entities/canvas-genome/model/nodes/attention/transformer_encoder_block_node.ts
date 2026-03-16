import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";

export class TransformerEncoderBlockNode extends BaseNode {
    private nHeads: number;
    private dFF: number; // Feed-forward hidden dimension
    private dropout: number;
    private activation: string;
    private normFirst: boolean; // Pre-norm vs post-norm

    constructor(
        nHeads: number = 4,
        dFF: number = 512,
        dropout: number = 0.1,
        activation: string = "gelu",
        normFirst: boolean = false
    ) {
        super();

        if (!(Number.isInteger(nHeads) && nHeads > 0)) {
            throw Error("nHeads must be a positive integer");
        }
        if (!(Number.isInteger(dFF) && dFF > 0)) {
            throw Error("dFF must be a positive integer");
        }
        if (!(dropout >= 0 && dropout <= 1)) {
            throw Error("dropout must be in [0, 1]");
        }

        this.nHeads = nHeads;
        this.dFF = dFF;
        this.dropout = dropout;
        this.activation = activation;
        this.normFirst = normFirst;

        // Input: [sequence_length, d_model]
        this.inputShape = new Array<number>(2);
    }

    protected CalculateOutputShape(): void {
        // TransformerEncoderBlock preserves shape: [sequence_length, d_model]
        this.outputShape = [...this.inputShape];
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                n_heads: this.nHeads,
                d_ff: this.dFF,
                dropout: this.dropout,
                activation: this.activation,
                norm_first: this.normFirst,
            },
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        const [seqLen, dModel] = this.inputShape;

        // Multi-head attention: 4 * (d_model^2 + d_model)
        const attentionParams = 4 * (dModel * dModel + dModel);

        // Position-wise feed-forward:
        // Linear(d_model, d_ff) + Linear(d_ff, d_model)
        // = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
        const ffParams =
            dModel * this.dFF +
            this.dFF +
            this.dFF * dModel +
            dModel;

        // LayerNorms: 2 * (d_model + d_model) for gamma, beta
        const normParams = 2 * (dModel + dModel);

        const flash = (attentionParams + ffParams + normParams) * dtype;

        // Memory:
        // Input: seq_len * d_model
        // Output: seq_len * d_model
        // Intermediate (FFN): seq_len * d_ff
        const ram =
            (2 * seqLen * dModel + seqLen * this.dFF +
                seqLen * seqLen) * // Attention scores
            dtype;

        // MACs:
        // MHA: 4 * seq_len * d_model * d_model
        // Attention aggregation: seq_len * seq_len * d_model
        // FFN: 2 * seq_len * d_model * d_ff
        const macs =
            4 * seqLen * dModel * dModel +
            seqLen * seqLen * dModel +
            2 * seqLen * dModel * this.dFF;

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        const dModel = this.inputShape[1];
        const maxHeads = Math.min(Math.floor(dModel / 2), 16);

        // Mutate n_heads: [1, min(d_model/2, 16)]
        if (Math.random() <= (mutation_options.get("teb_n_heads") || -1)) {
            this.nHeads = Math.max(1, Math.floor(Math.random() * maxHeads));
        }

        // Mutate d_ff: [d_model, d_model * 8]
        if (Math.random() <= (mutation_options.get("teb_d_ff") || -1)) {
            this.dFF = dModel + Math.floor(Math.random() * (dModel * 7));
        }

        // Mutate dropout: [0.0, 0.5]
        if (Math.random() <= (mutation_options.get("teb_dropout") || -1)) {
            this.dropout = Math.random() * 0.5;
        }

        // Mutate activation: choose from list
        if (Math.random() <= (mutation_options.get("teb_activation") || -1)) {
            const activations = ["gelu", "relu", "swish"];
            this.activation =
                activations[RandomizeInteger(0, activations.length - 1)];
        }

        // Mutate norm_first: toggle
        if (Math.random() <= (mutation_options.get("teb_norm_first") || -1)) {
            this.normFirst = !this.normFirst;
        }

        this.CalculateOutputShape();
    }

    public GetExpectedInputDimensions(): number | "any" {
        // Transformer encoder expects 2D input: [sequence_length, d_model]
        return 2;
    }

    public GetOutputDimensions(): number | "any" {
        // Outputs 2D: [sequence_length, d_model]
        return 2;
    }

    public GetNodeType = (): string => "TransformerEncoderBlock";

    protected _CloneImpl = (): BaseNode =>
        new TransformerEncoderBlockNode(
            this.nHeads,
            this.dFF,
            this.dropout,
            this.activation,
            this.normFirst
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

    GetDFF(): number {
        return this.dFF;
    }

    SetDFF(dFF: number): void {
        if (dFF > 0) {
            this.dFF = dFF;
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

    GetActivation(): string {
        return this.activation;
    }

    SetActivation(activation: string): void {
        this.activation = activation;
    }

    GetNormFirst(): boolean {
        return this.normFirst;
    }

    SetNormFirst(normFirst: boolean): void {
        this.normFirst = normFirst;
    }
}
