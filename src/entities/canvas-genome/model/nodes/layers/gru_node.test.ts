import { GRUNode } from "./gru_node";

describe("GRUNode", () => {
    it("should initialize with valid parameters", () => {
        const node = new GRUNode(128, "sigmoid", "tanh", true, true);
        expect(node.GetNodeType()).toBe("GRU");
        expect(node.GetHiddenUnits()).toBe(128);
        expect(node.GetGateActivation()).toBe("sigmoid");
        expect(node.GetHiddenActivation()).toBe("tanh");
        expect(node.GetUseBias()).toBe(true);
        expect(node.GetResetAfter()).toBe(true);
    });

    it("should use default parameters", () => {
        const node = new GRUNode(64);
        expect(node.GetHiddenUnits()).toBe(64);
        expect(node.GetGateActivation()).toBe("sigmoid");
        expect(node.GetHiddenActivation()).toBe("tanh");
        expect(node.GetUseBias()).toBe(true);
        expect(node.GetResetAfter()).toBe(true);
    });

    it("should throw error for invalid hidden units", () => {
        expect(() => new GRUNode(0)).toThrow();
        expect(() => new GRUNode(-10)).toThrow();
    });

    it("should calculate output shape correctly", () => {
        const node = new GRUNode(256, "sigmoid", "tanh", true, true);
        node["inputShape"] = [512, 64]; // [seq_len, input_features]

        node["CalculateOutputShape"]();

        // GRU preserves sequence length, outputs hidden_units per timestep
        expect(node["outputShape"]).toEqual([512, 256]);
    });

    it("should recalculate output shape when hidden units change", () => {
        const node = new GRUNode(128);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([512, 128]);

        node.SetHiddenUnits(256);

        expect(node["outputShape"]).toEqual([512, 256]);
    });

    it("should calculate resources correctly", () => {
        const node = new GRUNode(128, "sigmoid", "tanh", true, true);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4); // 4-byte float32

        // Flash (weights):
        // 3 gates * (input_size + hidden_units) * hidden_units + hidden_units
        // 3 * [(64 + 128) * 128 + 128] = 3 * 24576 = 73728
        const expectedFlash = 3 * ((64 + 128) * 128 + 128) * 4;
        expect(resources.flash).toBe(expectedFlash);

        // RAM:
        // input: 512 * 64 = 32768
        // output: 512 * 128 = 65536
        // hidden: 128
        // Total: 98432 * 4 = 393728
        const expectedRam = (512 * 64 + 512 * 128 + 128) * 4;
        expect(resources.ram).toBe(expectedRam);

        // MACs:
        // 3 * seq_len * (input_size + hidden_size) * hidden_size
        // 3 * 512 * 192 * 128 = 37748736
        const expectedMacs = 3 * 512 * (64 + 128) * 128;
        expect(resources.macs).toBe(expectedMacs);
    });

    it("should clone with all parameters preserved", () => {
        const original = new GRUNode(256, "sigmoid", "tanh", true, false);
        original["inputShape"] = [512, 64];
        original["CalculateOutputShape"]();

        const clone = original["_CloneImpl"]() as GRUNode;

        expect(clone.GetHiddenUnits()).toBe(256);
        expect(clone.GetGateActivation()).toBe("sigmoid");
        expect(clone.GetHiddenActivation()).toBe("tanh");
        expect(clone.GetUseBias()).toBe(true);
        expect(clone.GetResetAfter()).toBe(false);
    });

    it("should mutate hidden units", () => {
        const node = new GRUNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("gru_hidden_units", 1.0); // Always mutate

        node["Mutate"](options);

        expect(node.GetHiddenUnits()).toBeGreaterThan(0);
        expect(Number.isInteger(node.GetHiddenUnits())).toBe(true);
    });

    it("should mutate gate activation", () => {
        const node = new GRUNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("gru_gate_activation", 1.0);

        node["Mutate"](options);

        expect(["sigmoid", "relu", "leaky_relu"]).toContain(
            node.GetGateActivation()
        );
    });

    it("should mutate hidden activation", () => {
        const node = new GRUNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("gru_hidden_activation", 1.0);

        node["Mutate"](options);

        expect(["tanh", "sigmoid"]).toContain(node.GetHiddenActivation());
    });

    it("should mutate reset_after", () => {
        const node = new GRUNode(128, "sigmoid", "tanh", true, true);
        node["inputShape"] = [512, 64];

        const originalValue = node.GetResetAfter();
        const options = new Map<string, number>();
        options.set("gru_reset_after", 1.0);

        node["Mutate"](options);

        expect(node.GetResetAfter()).toBe(!originalValue);
    });

    it("should validate expected input dimensions", () => {
        const node = new GRUNode(128);
        expect(node.GetExpectedInputDimensions()).toBe(2);
    });

    it("should validate output dimensions", () => {
        const node = new GRUNode(128);
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("should not be a merge node", () => {
        const node = new GRUNode(128);
        expect(node.GetIsMerging()).toBe(false);
    });

    it("should generate correct JSON info", () => {
        const node = new GRUNode(128, "sigmoid", "tanh", true, true);
        const info = JSON.parse(node.GetInfo());

        expect(info.node).toBe("GRU");
        expect(info.params.hidden_units).toBe(128);
        expect(info.params.gate_activation).toBe("sigmoid");
        expect(info.params.hidden_activation).toBe("tanh");
        expect(info.params.use_bias).toBe(true);
        expect(info.params.reset_after).toBe(true);
    });

    it("should allow setting parameters", () => {
        const node = new GRUNode(128);
        node["inputShape"] = [512, 64];

        node.SetHiddenUnits(256);
        expect(node.GetHiddenUnits()).toBe(256);

        node.SetGateActivation("relu");
        expect(node.GetGateActivation()).toBe("relu");

        node.SetHiddenActivation("sigmoid");
        expect(node.GetHiddenActivation()).toBe("sigmoid");

        node.SetUseBias(false);
        expect(node.GetUseBias()).toBe(false);

        node.SetResetAfter(false);
        expect(node.GetResetAfter()).toBe(false);
    });

    it("should compare GRU with LSTM resource usage", () => {
        // GRU should use fewer parameters than LSTM (3 gates vs 4 gates)
        const gru = new GRUNode(128, "sigmoid", "tanh", true, true);
        gru["inputShape"] = [512, 64];
        gru["CalculateOutputShape"]();

        const gruResources = gru.GetResources(4);

        // Both with same hidden_units and input_size
        // GRU flash = 3 * gates, LSTM flash = 4 * gates
        // So GRU should be ~75% of LSTM
        expect(gruResources.flash).toBeLessThan(98304 * 4); // LSTM flash approximation
    });
});
