import { BaseNode, ResourceCriteria } from "../base_node";

export class Concat2DNode extends BaseNode {
    protected CalculateOutputShape(): void {
        if (this.previous.length == 0) {
            return
        }

        const h = this.inputShape[0]
        const w = this.inputShape[1]
        const c = this.previous.reduce((result, current) => result + current.GetOutputShape()[2], 0)

        this.outputShape = [h, w, c]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {}
        })
    }

    GetResources(_dtype: number): ResourceCriteria {
        return { flash: 0, ram: 0, macs: 0 }
    }

    protected Mutate(_mutation_options: Map<string, number>): void { }

    CheckCompability(node: BaseNode): Boolean {
        // Here `this` is linking to `node`.
        // We connect this -> node.
        // Therefore `node` needs to accept `this.outputShape`.
        // Wait, for Concat2D, it receives inputs. When checking if a node can connect to THIS:
        // Actually, in `store.ts`: `fromNode.CheckCompability(toNode)`.
        // So `this` = fromNode, `node` = toNode (Concat). 
        // Wait, CheckCompability is called ON the node we are connecting FROM?
        // Let's verify `store.ts`: `fromNode.node.CheckCompability(node as BaseNode)` 
        // Oh, if from is normal node, and to is Concat, Concat's CheckCompability isn't called!
        // But what if we connect from Concat to another node? Concat is checking the target node.

        // Either way, if `this` is Concat2DNode and it's calling CheckCompability(node),
        // it means we are connecting `Concat2D` -> `Node`.
        // Concat's output shape is 3D. The target node must accept 3D.
        // There are NO specific checks other than what the target node expects!
        // Wait! The target node's `CheckCompabilityDisconnected` or typical checks handle if IT accepts `Concat`.
        // But Concat2D itself when being the SOURCE doesn't have strict needs, it just needs target to be acyclic.

        // Wait, look at `Concat2DNode.CheckCompability` (old code):
        // this.inputShape[0] == node.GetOutputShape()[0] ...
        // This old logic was trying to check if the INCOMING node matches its shape.
        // But if `A.CheckCompability(B)` means A -> B, then this logic is checking if A's input = B's output, which is nonsense.
        // Concat2D just needs to ensure acyclic when connecting outward.

        // HOWEVER, what if user connects A -> Concat2D?
        // A calls A.CheckCompability(Concat2D).
        // Since A is e.g., Conv2D, it just checks if A is acyclic and Concat accepts 3D.
        return true && this.isAcyclic();
    }

    CheckCompabilityDisconnected(node: BaseNode): Boolean {
        return true;
    }

    protected AddPrev(node: BaseNode): void {
        if (this.previous.length == 0) {
            this.inputShape = [node.GetOutputShape()[0], node.GetOutputShape()[1], 0]
        }

        this.previous.push(node)
    }

    public GetNodeType = () => "Concat";

    public Clone = (): BaseNode => new Concat2DNode();

    public GetIsMerging = (): boolean => true;
} 