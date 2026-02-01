import { v4 } from "uuid"

export type ResourceCriteria = {
    flash: number
    ram: number
    macs: number
}

export abstract class BaseNode {
    public previous: BaseNode[] = []
    public next: BaseNode[] = []
    protected inputShape: number[] = []
    protected outputShape: number[] = []
    public id: string = v4();

    protected abstract CalculateOutputShape(): void
    abstract GetInfo(): String
    abstract GetResources(dtype: number): ResourceCriteria
    protected abstract Mutate(mutation_options: Map<string, number>): void
    abstract CheckCompability(node: BaseNode): Boolean

    private SetInputShape(newShape: number[]) {
        this.inputShape = newShape
    }

    public GetOutputShape = (): number[] => this.outputShape

    protected AddPrev(node: BaseNode) {
        this.previous.push(node)
        this.SetInputShape(node.outputShape)
        this.CalculateOutputShape()
        this.next.forEach(n => {
            n.SetInputShape(this.outputShape)
            n.CalculateOutputShape()
        })
    }

    public AddNext(node: BaseNode) {
        this.next.push(node)
        node.AddPrev(this)
    }

    protected RemovePrev(node: BaseNode) {
        this.previous = this.previous.filter(n => n != node);
    }

    public RemoveNext(node: BaseNode) {
        this.next = this.next = this.next.filter(n => n != node);
        node.RemovePrev(this);
    }

    public ClearAllConnections() {
        this.next.forEach(n => this.RemoveNext(n));
        this.previous.forEach(n => n.RemoveNext(this));
    }
}