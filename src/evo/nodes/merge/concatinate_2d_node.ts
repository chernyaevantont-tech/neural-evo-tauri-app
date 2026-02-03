import { BaseNode, ResourceCriteria } from "../base_node";

export class Concat2DNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.next.length == 0) {
            return
        }

        const h = this.inputShape[0]
        const w = this.inputShape[1]
        const c = this.next.reduce((result, current) => result + current.GetOutputShape()[2], 0)

        this.outputShape = [h, w, c]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {}
        })
    }

    GetResources(dtype: number): ResourceCriteria {
        return {flash: 0, ram: 0, macs: 0}
    }

    protected Mutate(mutation_options: Map<string, number>): void {}

    CheckCompability(node: BaseNode): Boolean {
        return this.inputShape.length == 0 ? true : 
            this.inputShape[0] == node.GetOutputShape()[0] &&
            this.inputShape[1] == node.GetOutputShape()[1]
    }
    
    protected AddPrev(node: BaseNode): void {
        if (this.previous.length == 0) {
            this.inputShape = [node.GetOutputShape()[0], node.GetOutputShape()[1], 0]
        }

        this.previous.push(node)
    }

    public GetNodeType = () => "Concat";

    public Clone = (): BaseNode  => new Concat2DNode();
} 