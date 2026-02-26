import { v4 } from "uuid"

export type ResourceCriteria = {
    flash: number
    ram: number
    macs: number
}

let _globalInnovationCounter = 0;
export const getNextInnovationId = () => ++_globalInnovationCounter;

export abstract class BaseNode {
    public previous: BaseNode[] = []
    public next: BaseNode[] = []
    protected inputShape: number[] = []
    protected outputShape: number[] = []
    public id: string = v4();
    public innovationId: number = getNextInnovationId();

    protected abstract CalculateOutputShape(): void
    abstract GetInfo(): string
    abstract GetResources(dtype: number): ResourceCriteria
    protected abstract Mutate(mutation_options: Map<string, number>): void
    protected abstract Mutate(mutation_options: Map<string, number>): void

    /** Returns how many dimensions this node expects its input tensor to have. */
    public abstract GetExpectedInputDimensions(): number | "any";

    /** Returns how many dimensions this node outputs. */
    public abstract GetOutputDimensions(): number | "any";

    /** Checks if this node can structurally accept a connection FROM `node`. */
    public CanAcceptConnectionFrom(node: BaseNode, isDisconnectedCheck: boolean = false): boolean {
        // Input nodes can NEVER accept incoming connections
        if (this.GetNodeType() === "Input") {
            return false;
        }

        // Output of source must match Expected Input of this (target)
        if (node.GetNodeType() === "Output") return false;

        // If this node isn't a merge node, it can only accept 1 incoming connection
        if (!this.GetIsMerging()) {
            if (isDisconnectedCheck) {
                // If checking for replacement, it might currently have 1 connection that will be replaced
                if (this.previous.length > 1) {
                    return false;
                }
            } else {
                // If it's a fresh connection, it must have 0 connections
                if (this.previous.length >= 1) {
                    return false;
                }
            }
        }

        // Output of source must match Expected Input of this (target)
        const sourceDims = node.GetOutputDimensions();
        const targetExpected = this.GetExpectedInputDimensions();

        if (sourceDims !== "any" && targetExpected !== "any") {
            if (sourceDims !== targetExpected) {
                return false;
            }
        }

        return true;
    }

    /** 
     * Verifies if connecting `this` -> `node` is valid.
     * `node` is the target being connected to. 
     */
    public CheckCompability(node: BaseNode): boolean {
        return node.CanAcceptConnectionFrom(this, false) && this.isAcyclic();
    }

    public CheckCompabilityDisconnected(node: BaseNode): boolean {
        // `isDisconnectedCheck = true` allows a target node that currently has 1 connection 
        // to return true, assuming that 1 connection will be swapped out.
        return node.CanAcceptConnectionFrom(this, true) && this.isAcyclic();
    }

    private SetInputShape(newShape: number[]) {
        this.inputShape = newShape
    }

    public GetOutputShape = (): number[] => this.outputShape;
    public GetInputShape = (): number[] => this.inputShape;

    public PropagateShapeUpdate(visited: Set<string> = new Set()): void {
        if (visited.has(this.id)) return;
        visited.add(this.id);

        this.CalculateOutputShape();
        this.next.forEach(n => {
            if (!n.GetIsMerging()) {
                n.SetInputShape(this.outputShape);
            }
            n.PropagateShapeUpdate(visited);
        });
    }

    protected AddPrev(node: BaseNode) {
        this.previous.push(node);
        if (!this.GetIsMerging()) {
            this.SetInputShape(node.GetOutputShape());
        }
        this.PropagateShapeUpdate();
    }

    public AddNext(node: BaseNode) {
        this.next.push(node)
        node.AddPrev(this)
    }

    protected RemovePrev(node: BaseNode) {
        this.previous = this.previous.filter(n => n.id !== node.id);
    }

    public RemoveNext(node: BaseNode) {
        this.next = this.next.filter(n => n.id !== node.id);
        node.RemovePrev(this);
    }

    public ClearAllConnections() {
        [...this.next].forEach(n => this.RemoveNext(n));
        [...this.previous].forEach(n => n.RemoveNext(this));
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

    public Clone(): BaseNode {
        const cloned = this._CloneImpl();
        cloned.innovationId = this.innovationId;
        cloned.id = v4();
        cloned.inputShape = [...this.inputShape];
        cloned.outputShape = [...this.outputShape];
        return cloned;
    }

    protected abstract _CloneImpl(): BaseNode;

    public abstract GetIsMerging(): boolean;
}