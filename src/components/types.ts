import { Genome } from "../evo/genome";
import { BaseNode } from "../evo/nodes/base_node";

export type Position = {
    x: number;
    y: number;
};

export type VisualNode = {
    id: string;
    node: BaseNode;
    position: Position;
    type: NodeType;
    genomeId: string | null;
};

export type VisualGenome = {
    id: string;
    genome: Genome;
}

export type Connection = {
    id: string;
    fromNodeId: string;
    toNodeId: string;
};

export type NodeType = 
    | "Input"
    | "Dense" 
    | "Conv2D" 
    | "Pooling"
    | "Flatten"
    | "Add"
    | "Concat2D";

export type NodeConfig = {
    type: NodeType;
    params: any;
};
