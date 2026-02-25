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
    abstract GetInfo(): string
    abstract GetResources(dtype: number): ResourceCriteria
    protected abstract Mutate(mutation_options: Map<string, number>): void
    abstract CheckCompability(node: BaseNode): Boolean
    abstract CheckCompabilityDisconnected(node: BaseNode): Boolean

    private SetInputShape(newShape: number[]) {
        this.inputShape = newShape
    }

    public GetOutputShape = (): number[] => this.outputShape;
    public GetInputShape = (): number[] => this.inputShape;

    protected AddPrev(node: BaseNode) {
        this.previous.push(node)
        this.SetInputShape(node.outputShape)
        this.CalculateOutputShape();
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
        [...this.next].forEach(n => this.RemoveNext(n));
        [...this.previous].forEach(n => n.RemovePrev(this));
    }

    public isAcyclic(): boolean {
        const visited = new Set<string>();
        const recursionStack = new Set<string>();

        const dfs = (node: BaseNode): boolean => {
            if (recursionStack.has(node.id)) {
                return false; // Цикл обнаружен
            }

            if (visited.has(node.id)) {
                return true; // Узел уже проверен
            }

            visited.add(node.id);
            recursionStack.add(node.id);

            for (const nextNode of node.next) {
                if (!dfs(nextNode)) {
                    return false;
                }
            }

            recursionStack.delete(node.id);
            return true;
        };

        return dfs(this);
    }

    public abstract GetNodeType(): string;

    public abstract Clone(): BaseNode;

    public abstract GetIsMerging(): boolean;
}