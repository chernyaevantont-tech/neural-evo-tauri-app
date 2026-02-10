import { useCallback, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { VisualGenome, VisualNode } from '../../shared/types';
import { BaseNode } from '../../evo/nodes/base_node';
import { Genome } from '../../evo/genome';
import { InputNode } from '../../evo/nodes/layers/input_node';
import { OutputNode } from '../../evo/nodes/layers/output_node';

export const createNewGenomeWithNode = (node: BaseNode): VisualGenome => {
  return {
    id: uuidv4(),
    genome: new Genome([node], [node]),
    isValid: false,
  };
};

export const updateGenomeValidity = (
  genomeNodes: VisualNode[],
  genome: Genome
): { isValid: boolean; inputNodes: BaseNode[]; outputNodes: BaseNode[] } => {
  let isValid = true;
  const inputNodes: BaseNode[] = [];
  const outputNodes: BaseNode[] = [];

  for (let visualNode of genomeNodes) {
    const node = visualNode.node;
    if (node.previous.length === 0) {
      inputNodes.push(node);
      if (!(node instanceof InputNode)) {
        isValid = false;
      }
    }
    if (node.next.length === 0) {
      outputNodes.push(node);
      if (!(node instanceof OutputNode)) {
        isValid = false;
      }
    }
  }

  genome.inputNodes = inputNodes;
  genome.outputNodes = outputNodes;

  return { isValid, inputNodes, outputNodes };
};
