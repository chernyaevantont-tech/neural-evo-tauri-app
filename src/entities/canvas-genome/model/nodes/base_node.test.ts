import { describe, it, expect } from 'vitest';
import { BaseNode, ResourceCriteria } from './base_node';

class DummyNode extends BaseNode {
    protected CalculateOutputShape(): void {
        this.outputShape = this.inputShape;
    }
    GetInfo(): string { return ""; }
    GetResources(): ResourceCriteria { return { flash: 0, ram: 0, macs: 0 }; }
    protected Mutate(): void { }
    CheckCompability(node: BaseNode): boolean { return true; }
    CheckCompabilityDisconnected(node: BaseNode): boolean { return true; }
    public GetNodeType = () => "Dummy";
    protected _CloneImpl = (): BaseNode => new DummyNode();
    public GetIsMerging = () => false;
    public GetExpectedInputDimensions(): number | "any" { return "any"; }
    public GetOutputDimensions(): number | "any" { return "any"; }
}

describe('BaseNode - Graph Topology', () => {
    it('should calculate acyclic corectly for a simple chain', () => {
        const n1 = new DummyNode();
        const n2 = new DummyNode();
        n1.AddNext(n2);

        expect(n1.isAcyclic()).toBe(true);
        expect(n2.isAcyclic()).toBe(true);
    });

    it('should detect cycles and return false', () => {
        const n1 = new DummyNode();
        const n2 = new DummyNode();
        const n3 = new DummyNode();

        n1.AddNext(n2);
        n2.AddNext(n3);
        n3.AddNext(n1); // Cycle

        expect(n1.isAcyclic()).toBe(false);
    });

    it('should connect and disconnect correctly', () => {
        const n1 = new DummyNode();
        const n2 = new DummyNode();

        n1.AddNext(n2);
        expect(n1.next.length).toBe(1);
        expect(n2.previous.length).toBe(1);

        n1.RemoveNext(n2);
        expect(n1.next.length).toBe(0);
        expect(n2.previous.length).toBe(0);
    });

    it('should clear all connections correctly', () => {
        const n1 = new DummyNode();
        const n2 = new DummyNode();
        const n3 = new DummyNode();

        n1.AddNext(n2);
        n2.AddNext(n3);

        expect(n2.previous.length).toBe(1);
        expect(n2.next.length).toBe(1);

        n2.ClearAllConnections();

        expect(n2.previous.length).toBe(0);
        expect(n2.next.length).toBe(0);
        expect(n1.next.length).toBe(0);
        expect(n3.previous.length).toBe(0);
    });
});
