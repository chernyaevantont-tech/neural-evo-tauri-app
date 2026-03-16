import { MultiHeadAttentionNode } from "./multihead_attention_node";

describe("MultiHeadAttentionNode", () => {
    it("should initialize with valid parameters", () => {
        const node = new MultiHeadAttentionNode(8, 0.1, false);
        expect(node.GetNodeType()).toBe("MultiHeadAttention");
        expect(node.GetNHeads()).toBe(8);
        expect(node.GetDropout()).toBeCloseTo(0.1);
        expect(node.GetQuietSoftmax()).toBe(false);
    });

    it("should use default parameters", () => {
        const node = new MultiHeadAttentionNode();
        expect(node.GetNHeads()).toBe(4);
        expect(node.GetDropout()).toBeCloseTo(0.1);
        expect(node.GetQuietSoftmax()).toBe(false);
    });

    it("should throw error for invalid n_heads", () => {
        expect(() => new MultiHeadAttentionNode(0)).toThrow();
        expect(() => new MultiHeadAttentionNode(-5)).toThrow();
    });

    it("should throw error for invalid dropout", () => {
        expect(() => new MultiHeadAttentionNode(4, -0.1)).toThrow();
        expect(() => new MultiHeadAttentionNode(4, 1.5)).toThrow();
    });

    it("should calculate output shape equal to input shape", () => {
        const node = new MultiHeadAttentionNode(4, 0.1, false);
        node["inputShape"] = [512, 64]; // [seq_len, d_model]

        node["CalculateOutputShape"]();

        // MHA preserves shape
        expect(node["outputShape"]).toEqual([512, 64]);
    });

    it("should recalculate output shape when input changes", () => {
        const node = new MultiHeadAttentionNode(4);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([512, 64]);

        node["inputShape"] = [256, 128];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([256, 128]);
    });

    it("should calculate resources correctly", () => {
        const node = new MultiHeadAttentionNode(4, 0.1, false);
        node["inputShape"] = [512, 64]; // [seq_len, d_model]
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4); // 4-byte float32

        // Flash (weights):
        // 4 projections: 4 * (64 * 64 + 64) = 4 * 4160 = 16640
        const expectedFlash = 4 * (64 * 64 + 64) * 4;
        expect(resources.flash).toBe(expectedFlash);

        // RAM:
        // input + output: 2 * 512 * 64 = 65536
        // attention scores: 512 * 512 = 262144
        // Total: (65536 + 262144) * 4 = 1150080
        const expectedRam = (2 * 512 * 64 + 512 * 512) * 4;
        expect(resources.ram).toBe(expectedRam);

        // MACs:
        // 4 projections: 4 * 512 * 64 * 64 = 8388608
        // Attention: 512 * 512 * 64 = 16777216
        // Total: 25165824
        const expectedMacs = 4 * 512 * 64 * 64 + 512 * 512 * 64;
        expect(resources.macs).toBe(expectedMacs);
    });

    it("should clone with all parameters preserved", () => {
        const original = new MultiHeadAttentionNode(8, 0.2, true);
        original["inputShape"] = [512, 64];
        original["CalculateOutputShape"]();

        const clone = original["_CloneImpl"]() as MultiHeadAttentionNode;

        expect(clone.GetNHeads()).toBe(8);
        expect(clone.GetDropout()).toBeCloseTo(0.2);
        expect(clone.GetQuietSoftmax()).toBe(true);
    });

    it("should mutate n_heads", () => {
        const node = new MultiHeadAttentionNode(4);
        node["inputShape"] = [512, 64]; // d_model = 64

        const options = new Map<string, number>();
        options.set("mha_n_heads", 1.0); // Always mutate

        node["Mutate"](options);

        expect(node.GetNHeads()).toBeGreaterThanOrEqual(1);
        expect(node.GetNHeads()).toBeLessThanOrEqual(32);
    });

    it("should mutate dropout", () => {
        const node = new MultiHeadAttentionNode(4);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("mha_dropout", 1.0);

        node["Mutate"](options);

        expect(node.GetDropout()).toBeGreaterThanOrEqual(0);
        expect(node.GetDropout()).toBeLessThanOrEqual(0.5);
    });

    it("should mutate quiet_softmax", () => {
        const node = new MultiHeadAttentionNode(4, 0.1, false);
        node["inputShape"] = [512, 64];

        const originalValue = node.GetQuietSoftmax();
        const options = new Map<string, number>();
        options.set("mha_quiet_softmax", 1.0);

        node["Mutate"](options);

        expect(node.GetQuietSoftmax()).toBe(!originalValue);
    });

    it("should validate expected input dimensions", () => {
        const node = new MultiHeadAttentionNode();
        expect(node.GetExpectedInputDimensions()).toBe(2);
    });

    it("should validate output dimensions", () => {
        const node = new MultiHeadAttentionNode();
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("should not be a merge node", () => {
        const node = new MultiHeadAttentionNode();
        expect(node.GetIsMerging()).toBe(false);
    });

    it("should generate correct JSON info", () => {
        const node = new MultiHeadAttentionNode(8, 0.2, true);
        const info = JSON.parse(node.GetInfo());

        expect(info.node).toBe("MultiHeadAttention");
        expect(info.params.n_heads).toBe(8);
        expect(info.params.dropout).toBeCloseTo(0.2);
        expect(info.params.quiet_softmax).toBe(true);
    });

    it("should allow setting parameters", () => {
        const node = new MultiHeadAttentionNode();
        node["inputShape"] = [512, 64];

        node.SetNHeads(8);
        expect(node.GetNHeads()).toBe(8);

        node.SetDropout(0.25);
        expect(node.GetDropout()).toBeCloseTo(0.25);

        node.SetQuietSoftmax(true);
        expect(node.GetQuietSoftmax()).toBe(true);
    });

    it("should constrain n_heads based on d_model", () => {
        const node = new MultiHeadAttentionNode(4, 0.1, false);
        node["inputShape"] = [512, 32]; // d_model = 32, max_heads = min(32/2, 16) = 16

        const options = new Map<string, number>();
        options.set("mha_n_heads", 1.0);

        for (let i = 0; i < 10; i++) {
            node["Mutate"](options);
            expect(node.GetNHeads()).toBeLessThanOrEqual(16);
        }
    });

    it("should validate dropout range for setter", () => {
        const node = new MultiHeadAttentionNode();

        node.SetDropout(-0.1);
        expect(node.GetDropout()).toBeCloseTo(0.1); // Should not change

        node.SetDropout(0.5);
        expect(node.GetDropout()).toBeCloseTo(0.5); // Valid

        node.SetDropout(1.5);
        expect(node.GetDropout()).toBeCloseTo(0.5); // Should not change
    });
});
