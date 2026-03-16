import { TransformerEncoderBlockNode } from "./transformer_encoder_block_node";

describe("TransformerEncoderBlockNode", () => {
    it("should initialize with valid parameters", () => {
        const node = new TransformerEncoderBlockNode(8, 2048, 0.1, "gelu", false);
        expect(node.GetNodeType()).toBe("TransformerEncoderBlock");
        expect(node.GetNHeads()).toBe(8);
        expect(node.GetDFF()).toBe(2048);
        expect(node.GetDropout()).toBeCloseTo(0.1);
        expect(node.GetActivation()).toBe("gelu");
        expect(node.GetNormFirst()).toBe(false);
    });

    it("should use default parameters", () => {
        const node = new TransformerEncoderBlockNode();
        expect(node.GetNHeads()).toBe(4);
        expect(node.GetDFF()).toBe(512);
        expect(node.GetDropout()).toBeCloseTo(0.1);
        expect(node.GetActivation()).toBe("gelu");
        expect(node.GetNormFirst()).toBe(false);
    });

    it("should throw error for invalid n_heads", () => {
        expect(() => new TransformerEncoderBlockNode(0)).toThrow();
        expect(() => new TransformerEncoderBlockNode(-5)).toThrow();
    });

    it("should throw error for invalid d_ff", () => {
        expect(() => new TransformerEncoderBlockNode(4, 0)).toThrow();
        expect(() => new TransformerEncoderBlockNode(4, -512)).toThrow();
    });

    it("should throw error for invalid dropout", () => {
        expect(() => new TransformerEncoderBlockNode(4, 512, -0.1)).toThrow();
        expect(() => new TransformerEncoderBlockNode(4, 512, 1.5)).toThrow();
    });

    it("should calculate output shape equal to input shape", () => {
        const node = new TransformerEncoderBlockNode(4, 512, 0.1, "gelu", false);
        node["inputShape"] = [512, 64]; // [seq_len, d_model]

        node["CalculateOutputShape"]();

        // Transformer encoder preserves shape
        expect(node["outputShape"]).toEqual([512, 64]);
    });

    it("should recalculate output shape when input changes", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([512, 64]);

        node["inputShape"] = [256, 128];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([256, 128]);
    });

    it("should calculate resources correctly", () => {
        const node = new TransformerEncoderBlockNode(4, 256, 0.1, "gelu", false);
        node["inputShape"] = [512, 64]; // [seq_len, d_model]
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4); // 4-byte float32

        const dModel = 64;
        const seqLen = 512;
        const dFF = 256;

        // Flash (weights):
        // MHA: 4 * (64*64 + 64) = 4 * 4160 = 16640
        // FFN: 64*256 + 256 + 256*64 + 64 = 32064
        // LayerNorm: 2 * (64 + 64) = 256
        // Total: 48960 * 4 = 195840
        const mhaParams = 4 * (dModel * dModel + dModel);
        const ffnParams = dModel * dFF + dFF + dFF * dModel + dModel;
        const normParams = 2 * (dModel + dModel);
        const expectedFlash = (mhaParams + ffnParams + normParams) * 4;
        expect(resources.flash).toBe(expectedFlash);

        // RAM:
        // input: 512 * 64 = 32768
        // output: 512 * 64 = 32768
        // intermediate: 512 * 256 = 131072
        // attention scores: 512 * 512 = 262144
        // Total: 458752 * 4 = 1835008
        const expectedRam =
            (2 * seqLen * dModel + seqLen * dFF + seqLen * seqLen) * 4;
        expect(resources.ram).toBe(expectedRam);

        // MACs:
        // MHA: 4 * seq_len * d_model * d_model = 4 * 512 * 64 * 64 = 8388608
        // Attention aggregation: 512 * 512 * 64 = 16777216
        // FFN: 2 * 512 * 64 * 256 = 16777216
        // Total: 41943040
        const expectedMacs =
            4 * seqLen * dModel * dModel +
            seqLen * seqLen * dModel +
            2 * seqLen * dModel * dFF;
        expect(resources.macs).toBe(expectedMacs);
    });

    it("should clone with all parameters preserved", () => {
        const original = new TransformerEncoderBlockNode(
            8,
            2048,
            0.2,
            "relu",
            true
        );
        original["inputShape"] = [512, 64];
        original["CalculateOutputShape"]();

        const clone = original["_CloneImpl"]() as TransformerEncoderBlockNode;

        expect(clone.GetNHeads()).toBe(8);
        expect(clone.GetDFF()).toBe(2048);
        expect(clone.GetDropout()).toBeCloseTo(0.2);
        expect(clone.GetActivation()).toBe("relu");
        expect(clone.GetNormFirst()).toBe(true);
    });

    it("should mutate n_heads", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64]; // d_model = 64

        const options = new Map<string, number>();
        options.set("teb_n_heads", 1.0);

        node["Mutate"](options);

        expect(node.GetNHeads()).toBeGreaterThanOrEqual(1);
        expect(node.GetNHeads()).toBeLessThanOrEqual(32);
    });

    it("should mutate d_ff", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64]; // d_model = 64

        const dModel = 64;
        const options = new Map<string, number>();
        options.set("teb_d_ff", 1.0);

        node["Mutate"](options);

        expect(node.GetDFF()).toBeGreaterThanOrEqual(dModel);
        expect(node.GetDFF()).toBeLessThanOrEqual(dModel * 8);
    });

    it("should mutate dropout", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("teb_dropout", 1.0);

        node["Mutate"](options);

        expect(node.GetDropout()).toBeGreaterThanOrEqual(0);
        expect(node.GetDropout()).toBeLessThanOrEqual(0.5);
    });

    it("should mutate activation", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("teb_activation", 1.0);

        node["Mutate"](options);

        expect(["gelu", "relu", "swish"]).toContain(node.GetActivation());
    });

    it("should mutate norm_first", () => {
        const node = new TransformerEncoderBlockNode(4, 512, 0.1, "gelu", false);
        node["inputShape"] = [512, 64];

        const originalValue = node.GetNormFirst();
        const options = new Map<string, number>();
        options.set("teb_norm_first", 1.0);

        node["Mutate"](options);

        expect(node.GetNormFirst()).toBe(!originalValue);
    });

    it("should validate expected input dimensions", () => {
        const node = new TransformerEncoderBlockNode();
        expect(node.GetExpectedInputDimensions()).toBe(2);
    });

    it("should validate output dimensions", () => {
        const node = new TransformerEncoderBlockNode();
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("should not be a merge node", () => {
        const node = new TransformerEncoderBlockNode();
        expect(node.GetIsMerging()).toBe(false);
    });

    it("should generate correct JSON info", () => {
        const node = new TransformerEncoderBlockNode(8, 2048, 0.2, "relu", true);
        const info = JSON.parse(node.GetInfo());

        expect(info.node).toBe("TransformerEncoderBlock");
        expect(info.params.n_heads).toBe(8);
        expect(info.params.d_ff).toBe(2048);
        expect(info.params.dropout).toBeCloseTo(0.2);
        expect(info.params.activation).toBe("relu");
        expect(info.params.norm_first).toBe(true);
    });

    it("should allow setting parameters", () => {
        const node = new TransformerEncoderBlockNode();
        node["inputShape"] = [512, 64];

        node.SetNHeads(8);
        expect(node.GetNHeads()).toBe(8);

        node.SetDFF(2048);
        expect(node.GetDFF()).toBe(2048);

        node.SetDropout(0.25);
        expect(node.GetDropout()).toBeCloseTo(0.25);

        node.SetActivation("relu");
        expect(node.GetActivation()).toBe("relu");

        node.SetNormFirst(true);
        expect(node.GetNormFirst()).toBe(true);
    });

    it("should compute more resources than simple MHA", () => {
        // TransformerEncoderBlock includes both MHA and FFN
        // So it should use more resources than MHA alone

        const node = new TransformerEncoderBlockNode(4, 256, 0.1, "gelu", false);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4);

        // Should have significant flash/RAM/MACs for both components
        expect(resources.flash).toBeGreaterThan(16640 * 4); // More than MHA alone
        expect(resources.ram).toBeGreaterThan(100000); // Significant memory
        expect(resources.macs).toBeGreaterThan(1000000); // Significant computation
    });

    it("should validate dropout range for setter", () => {
        const node = new TransformerEncoderBlockNode();

        node.SetDropout(-0.1);
        expect(node.GetDropout()).toBeCloseTo(0.1); // Should not change

        node.SetDropout(0.3);
        expect(node.GetDropout()).toBeCloseTo(0.3); // Valid

        node.SetDropout(1.5);
        expect(node.GetDropout()).toBeCloseTo(0.3); // Should not change
    });
});
