import { BaseNode, ResourceCriteria } from "../base_node";

function normalizeImageShapeToHwc(shape: number[]): number[] {
    if (shape.length !== 3) {
        return [...shape];
    }

    const [a, b, c] = shape;
    const firstIsChannel = a <= 4 && b > 4 && c > 4;
    const lastIsChannel = c <= 4 && a > 4 && b > 4;

    if (firstIsChannel && !lastIsChannel) {
        return [b, c, a];
    }

    return [...shape];
}

export class InputNode extends BaseNode {

    constructor(outputShape: number[]) {
        super()
        this.outputShape = normalizeImageShapeToHwc(outputShape)
    }

    protected CalculateOutputShape(): void { }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                output_shape: this.outputShape
            }
        });
    }

    GetResources(_dtype: Number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }
    }
    protected Mutate(_mutation_options: Map<string, number>): void { }

    public GetExpectedInputDimensions(): number | "any" {
        return "any";
    }

    public GetOutputDimensions(): number | "any" {
        return this.outputShape.length;
    }

    public GetNodeType = (): string => "Input";

    protected _CloneImpl = (): BaseNode => new InputNode(
        [...this.outputShape],
    );

    public GetIsMerging = (): boolean => false;

    public override CanHaveInput = (): boolean => false;
}