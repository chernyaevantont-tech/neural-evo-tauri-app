import { Genome } from "../evo/genome";
import { BaseNode } from "../evo/nodes/base_node";

export type Position = {
    x: number;
    y: number;
};

export type VisualNode = {
    node: BaseNode;
    position: Position;
    genomeId: string;
};

export type VisualGenome = {
    id: string;
    genome: Genome;
    isValid: boolean;
}

export type NewNodeDto = {
    node: BaseNode;
}

export type Connection = {
    id: string;
    fromNodeId: string;
    toNodeId: string;
};

export type NodeConfig = {
    params: any;
};
