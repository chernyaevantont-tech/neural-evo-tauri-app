# Network Canvas: Data Consistency and Compatibility Rules

This document describes how data consistency is maintained in the `canvas-genome` architecture, how node connections are validated, and how topological integrity is preserved.

## 1. Data Consistency Management (`store.ts`)

The application uses Zustand + Immer for state management (`useCanvasGenomeStore`). The core entities are:
*   `nodes`: A Map linking a unique `nodeId` to a `VisualNode` (which wraps `BaseNode`).
*   `connections`: A Map of connection objects `{ id, fromNodeId, toNodeId }`.
*   `genomes`: A Map of `VisualGenome` objects (a valid neural network graph).
*   `genomeNode`: A Map linking `genomeId` to an array of `VisualNode` objects belonging to that genome.

### Node Operations and Consistency Guarantees
*   **Adding a Node (`addNode`)**: Creates an isolated `VisualNode` and assigns it a completely new, unique `genomeId`.
*   **Editing a Node (`editNode`)**:
    *   Finds all incoming (`inConnections`) and outgoing (`outConnections`) connections.
    *   Internally disconnects the old node (using `RemoveNext()`).
    *   Creates a new instance of the node with new parameters.
    *   Attempts to reconnect all previous incoming and outgoing nodes to the *new* node instance using `CheckCompability`.
    *   If any reconnection fails (throws an Error), the entire Zustand `set` operation is aborted (thanks to Immer/Zustand structure), safely reverting the state.
*   **Connecting Nodes (`connectNodes`)**:
    *   Verifies that the `fromNode` and `toNode` belong to different `.next`/`.previous` relations (avoids duplicating existing edges).
    *   Calls `fromNode.CheckCompability(toNode)`. If `true`, adds the connection.
    *   Combines the two previously separate genomes into a single genome by updating `genomeNode` maps and reassigning `genomeId` on all connected nodes.
*   **Deleting a Node (`deleteNode`)**:
    *   Removes all visual `connections` involving the deleted node.
    *   Calls `ClearAllConnections()` on the `BaseNode` instance to sever internal graph links (`.next` and `.previous` arrays).
    *   *Graph Splitting (Crucial)*: Because deleting a node can split one genome into multiple disconnected subgraphs, the store runs a Breadth-First Search (BFS) starting from every node previously connected to the deleted node. Each distinct BFS island gets assigned a new `genomeId`.
*   **Deleting a Connection (`deleteConnection`)**:
    *   Similar to `deleteNode`, it calls `RemoveNext` between the two nodes.
    *   Runs BFS from both the source and target node of the severed connection. If they don't discover each other, they are split into two separate genomes with new IDs.

## 2. Shape Propagation (`CalculateOutputShape`)

When a node is connected (`AddNext` / `AddPrev`), it calls `CalculateOutputShape()`.
1.  **Forward Propagation**: When `nodeA.AddNext(nodeB)` is called, `nodeB.SetInputShape(nodeA.outputShape)` is triggered.
2.  **Cascade**: `nodeB` calls its own `CalculateOutputShape()`.
3.  **Recursive Update**: `nodeB` then iterates over all its `.next` nodes, updating their input shapes and commanding them to `CalculateOutputShape()`, propagating the change entirely down the graph until the output nodes.

### Merge Node specific rules:
*   **AddNode**: Assumes the output shape is identical to the input shape. *Bug spotted: `AddNode.CalculateOutputShape()` just copies `this.inputShape`. It does not verify if multiple incoming connections have matching shapes during shape calculation, meaning the topological shape calculation depends solely on the very last connected node.*
*   **Concat2DNode**: Sets `c` (channels) to the sum of all incoming nodes' `c` channels. *Bug spotted: `Concat2DNode` calculates its output shape based on `this.next.reduce(...)` instead of `this.previous.reduce(...)`. This means a concat node calculates its *output* channel depth based on its *children*, which violates forward-pass logic.*

## 3. Node Compatibility Rules (`CheckCompability`)

Before a connection is made (either by user Shift+Click or during node editing/reconstruction), `fromNode.CheckCompability(toNode)` is called.

**General Rule for Standard Nodes (`Dense`, `Conv2D`, `Pooling`, `Flatten`)**:
1.  **Acyclicity**: `this.isAcyclic()` is usually checked (but often placed incorrectly in the boolean chain).
2.  **Fan-in Restriction**: Standard layers usually only accept 1 input. Therefore, `node.previous.length == 0` is checked (meaning the target node cannot already have an input).
3.  **Shape Constraints**:
    *   1D operations (`Dense`) check if `node.GetInputShape().length == 1`.
    *   3D operations (`Conv2D`, `Pooling`) check if `node.GetInputShape().length == 3`.

**Merge Node Constraints**:
*   **AddNode**: Target rules are loose. Target must match `this.inputShape[index]`.
*   **Concat2DNode**: `this.inputShape[0] == node.GetOutputShape()[0] && this.inputShape[1] == node.GetOutputShape()[1]`. Checks if spatial dimensions (H, W) match. *Bug spotted: Concat2DNode checks `this.inputShape` against `node.GetOutputShape()`, which is backward. It should check `this.outputShape` against `node.GetInputShape()` or incoming node's output shape contextually.*

