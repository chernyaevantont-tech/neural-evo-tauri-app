import { LSTMNode } from "./lstm_node";

describe("LSTMNode", () => {
    it("should initialize with valid parameters", () => {
        const node = new LSTMNode(128, "sigmoid", "tanh", "tanh", true);
        expect(node.GetNodeType()).toBe("LSTM");
        expect(node.GetHiddenUnits()).toBe(128);
        expect(node.GetGateActivation()).toBe("sigmoid");
        expect(node.GetCellActivation()).toBe("tanh");
        expect(node.GetHiddenActivation()).toBe("tanh");
        expect(node.GetUseBias()).toBe(true);
    });

    it("should use default parameters", () => {
        const node = new LSTMNode(64);
        expect(node.GetHiddenUnits()).toBe(64);
        expect(node.GetGateActivation()).toBe("sigmoid");
        expect(node.GetCellActivation()).toBe("tanh");
        expect(node.GetHiddenActivation()).toBe("tanh");
        expect(node.GetUseBias()).toBe(true);
    });

    it("should throw error for invalid hidden units", () => {
        expect(() => new LSTMNode(0)).toThrow();
        expect(() => new LSTMNode(-10)).toThrow();
    });

    it("should calculate output shape correctly", () => {
        const node = new LSTMNode(256, "sigmoid", "tanh", "tanh", true);
        node["inputShape"] = [512, 64]; // [seq_len, input_features]

        node["CalculateOutputShape"]();

        // LSTM preserves sequence length, outputs hidden_units per timestep
        expect(node["outputShape"]).toEqual([512, 256]);
    });

    it("should recalculate output shape when hidden units change", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        expect(node["outputShape"]).toEqual([512, 128]);

        node.SetHiddenUnits(256);

        expect(node["outputShape"]).toEqual([512, 256]);
    });

    it("should calculate resources correctly", () => {
        const node = new LSTMNode(128, "sigmoid", "tanh", "tanh", true);
        node["inputShape"] = [512, 64];
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4); // 4-byte float32

        // Flash (weights):
        // 4 gates * (input_size + hidden_units) * hidden_units + hidden_units
        // 4 * [(64 + 128) * 128 + 128] = 4 * 24576 = 98304
        const expectedFlash = 4 * ((64 + 128) * 128 + 128) * 4;
        expect(resources.flash).toBe(expectedFlash);

        // RAM:
        // input: 512 * 64 = 32768
        // output: 512 * 128 = 65536
        // hidden + cell: 128 + 128 = 256
        // Total: 98560 * 4 = 394240
        const expectedRam = (512 * 64 + 512 * 128 + 2 * 128) * 4;
        expect(resources.ram).toBe(expectedRam);

        // MACs:
        // 4 * seq_len * (input_size + hidden_size) * hidden_size
        // 4 * 512 * 192 * 128 = 50331648
        const expectedMacs = 4 * 512 * (64 + 128) * 128;
        expect(resources.macs).toBe(expectedMacs);
    });

    it("should clone with all parameters preserved", () => {
        const original = new LSTMNode(256, "sigmoid", "tanh", "tanh", true);
        original["inputShape"] = [512, 64];
        original["CalculateOutputShape"]();

        const clone = original["_CloneImpl"]() as LSTMNode;

        expect(clone.GetHiddenUnits()).toBe(256);
        expect(clone.GetGateActivation()).toBe("sigmoid");
        expect(clone.GetCellActivation()).toBe("tanh");
        expect(clone.GetHiddenActivation()).toBe("tanh");
        expect(clone.GetUseBias()).toBe(true);
    });

    it("should mutate hidden units", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("lstm_hidden_units", 1.0); // Always mutate

        node["Mutate"](options);

        expect(node.GetHiddenUnits()).toBeGreaterThan(0);
        expect(Number.isInteger(node.GetHiddenUnits())).toBe(true);
    });

    it("should mutate gate activation", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("lstm_gate_activation", 1.0);

        node["Mutate"](options);

        expect(["sigmoid", "relu", "leaky_relu"]).toContain(
            node.GetGateActivation()
        );
    });

    it("should mutate cell activation", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("lstm_cell_activation", 1.0);

        node["Mutate"](options);

        expect(["tanh", "sigmoid"]).toContain(node.GetCellActivation());
    });

    it("should mutate hidden activation", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];

        const options = new Map<string, number>();
        options.set("lstm_hidden_activation", 1.0);

        node["Mutate"](options);

        expect(["tanh", "sigmoid"]).toContain(node.GetHiddenActivation());
    });

    it("should validate expected input dimensions", () => {
        const node = new LSTMNode(128);
        expect(node.GetExpectedInputDimensions()).toBe(2);
    });

    it("should validate output dimensions", () => {
        const node = new LSTMNode(128);
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("should not be a merge node", () => {
        const node = new LSTMNode(128);
        expect(node.GetIsMerging()).toBe(false);
    });

    it("should generate correct JSON info", () => {
        const node = new LSTMNode(128, "sigmoid", "tanh", "tanh", true);
        const info = JSON.parse(node.GetInfo());

        expect(info.node).toBe("LSTM");
        expect(info.params.hidden_units).toBe(128);
        expect(info.params.gate_activation).toBe("sigmoid");
        expect(info.params.cell_activation).toBe("tanh");
        expect(info.params.hidden_activation).toBe("tanh");
        expect(info.params.use_bias).toBe(true);
    });

    it("should allow setting parameters", () => {
        const node = new LSTMNode(128);
        node["inputShape"] = [512, 64];

        node.SetHiddenUnits(256);
        expect(node.GetHiddenUnits()).toBe(256);

        node.SetGateActivation("relu");
        expect(node.GetGateActivation()).toBe("relu");

        node.SetCellActivation("sigmoid");
        expect(node.GetCellActivation()).toBe("sigmoid");

        node.SetHiddenActivation("sigmoid");
        expect(node.GetHiddenActivation()).toBe("sigmoid");

        node.SetUseBias(false);
        expect(node.GetUseBias()).toBe(false);
    });
});
