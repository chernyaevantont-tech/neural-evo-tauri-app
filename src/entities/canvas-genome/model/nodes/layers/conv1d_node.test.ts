import { Conv1DNode } from "./conv1d_node";

describe("Conv1DNode", () => {
    it("should initialize with valid parameters", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        expect(node.GetNodeType()).toBe("Conv1D");
        expect(node.GetFilters()).toBe(32);
        expect(node.GetKernelSize()).toBe(3);
    });

    it("should throw error for invalid filters", () => {
        expect(() => new Conv1DNode(0, 3, 1, 1, 1, true)).toThrow();
        expect(() => new Conv1DNode(-5, 3, 1, 1, 1, true)).toThrow();
    });

    it("should throw error for invalid kernel size", () => {
        expect(() => new Conv1DNode(32, 0, 1, 1, 1, true)).toThrow();
        expect(() => new Conv1DNode(32, -3, 1, 1, 1, true)).toThrow();
    });

    it("should throw error for invalid stride", () => {
        expect(() => new Conv1DNode(32, 3, 0, 1, 1, true)).toThrow();
        expect(() => new Conv1DNode(32, 3, -1, 1, 1, true)).toThrow();
    });

    it("should throw error for invalid dilation", () => {
        expect(() => new Conv1DNode(32, 3, 1, 1, 0, true)).toThrow();
    });

    it("should calculate output shape correctly", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16]; // [seq_len, input_channels]

        node["CalculateOutputShape"]();

        // L_out = floor((512 + 2*1 - 1*(3-1) - 1) / 1 + 1) = floor(512) = 512
        expect(node["outputShape"]).toEqual([512, 32]);
    });

    it("should calculate output shape with stride > 1", () => {
        const node = new Conv1DNode(64, 3, 2, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        node["CalculateOutputShape"]();

        // L_out = floor((512 + 2*1 - 1*(3-1) - 1) / 2 + 1) = floor(256.5) = 256
        expect(node["outputShape"]).toEqual([256, 64]);
    });

    it("should calculate output shape with dilation > 1", () => {
        const node = new Conv1DNode(32, 3, 1, 0, 2, true, "relu");
        node["inputShape"] = [512, 16];

        node["CalculateOutputShape"]();

        // L_out = floor((512 + 0 - 2*(3-1) - 1) / 1 + 1) = floor(507 + 1) = 508
        expect(node["outputShape"]).toEqual([508, 32]);
    });

    it("should calculate resources correctly", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];
        node["CalculateOutputShape"]();

        const resources = node.GetResources(4); // 4-byte float32

        // Flash (weights): 32 * (3 * 16 + 1) * 4 = 32 * 49 * 4 = 6272
        expect(resources.flash).toBe(32 * (3 * 16 + 1) * 4);

        // RAM: (512 * 16 + 512 * 32) * 4 = (8192 + 16384) * 4 = 98304
        expect(resources.ram).toBe((512 * 16 + 512 * 32) * 4);

        // MACs: 512 * 32 * 3 * 16 = 786432
        expect(resources.macs).toBe(512 * 32 * 3 * 16);
    });

    it("should clone with all parameters preserved", () => {
        const original = new Conv1DNode(64, 5, 2, 1, 2, true, "relu");
        original["inputShape"] = [512, 32];
        original["CalculateOutputShape"]();

        const clone = original["_CloneImpl"]() as Conv1DNode;

        expect(clone.GetFilters()).toBe(64);
        expect(clone.GetKernelSize()).toBe(5);
        expect(clone.GetStride()).toBe(2);
        expect(clone.GetPadding()).toBe(1);
        expect(clone.GetDilation()).toBe(2);
        expect(clone.GetUseBias()).toBe(true);
        expect(clone.GetActivation()).toBe("relu");
    });

    it("should mutate filters", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        const options = new Map<string, number>();
        options.set("conv1d_filters", 1.0); // Always mutate

        node["Mutate"](options);

        expect(node.GetFilters()).toBeGreaterThan(0);
        expect(Number.isInteger(node.GetFilters())).toBe(true);
    });

    it("should mutate kernel size", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        const options = new Map<string, number>();
        options.set("conv1d_kernel_size", 1.0);

        node["Mutate"](options);

        // Kernel size should be odd and in range [3, 11]
        expect([3, 5, 7, 9, 11]).toContain(node.GetKernelSize());
    });

    it("should mutate stride", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        const options = new Map<string, number>();
        options.set("conv1d_stride", 1.0);

        node["Mutate"](options);

        expect([1, 2]).toContain(node.GetStride());
    });

    it("should mutate activation", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        const options = new Map<string, number>();
        options.set("conv1d_activation", 1.0);

        node["Mutate"](options);

        expect(["relu", "leaky_relu", "sigmoid", "tanh", "linear"]).toContain(
            node.GetActivation()
        );
    });

    it("should validate expected input dimensions", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        expect(node.GetExpectedInputDimensions()).toBe(2);
    });

    it("should validate output dimensions", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("should not be a merge node", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        expect(node.GetIsMerging()).toBe(false);
    });

    it("should generate correct JSON info", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        const info = JSON.parse(node.GetInfo());

        expect(info.node).toBe("Conv1D");
        expect(info.params.filters).toBe(32);
        expect(info.params.kernel_size).toBe(3);
        expect(info.params.stride).toBe(1);
        expect(info.params.padding).toBe(1);
        expect(info.params.dilation).toBe(1);
        expect(info.params.use_bias).toBe(true);
        expect(info.params.activation).toBe("relu");
    });

    it("should allow setting parameters", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];

        node.SetFilters(64);
        expect(node.GetFilters()).toBe(64);

        node.SetKernelSize(5);
        expect(node.GetKernelSize()).toBe(5);

        node.SetStride(2);
        expect(node.GetStride()).toBe(2);

        node.SetPadding(2);
        expect(node.GetPadding()).toBe(2);

        node.SetDilation(4);
        expect(node.GetDilation()).toBe(4);

        node.SetUseBias(false);
        expect(node.GetUseBias()).toBe(false);

        node.SetActivation("sigmoid");
        expect(node.GetActivation()).toBe("sigmoid");
    });

    it("should recalculate output shape when parameters change", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1, true, "relu");
        node["inputShape"] = [512, 16];
        node["CalculateOutputShape"]();

        const originalOutput = [...node["outputShape"]];
        expect(originalOutput).toEqual([512, 32]);

        node.SetStride(2);

        const newOutput = node["outputShape"];
        expect(newOutput[0]).toBeLessThan(originalOutput[0]);
    });
});
