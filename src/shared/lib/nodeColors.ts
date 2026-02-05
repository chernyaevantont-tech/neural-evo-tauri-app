import { theme } from './theme';

export const getNodeColor = (type: string): string => {
  switch (type) {
    case 'Input': return theme.colors.node.input;
    case 'Dense': return theme.colors.node.dense;
    case 'Conv2D': return theme.colors.node.conv2d;
    case 'Pooling': return theme.colors.node.pooling;
    case 'Flatten': return theme.colors.node.flatten;
    case 'Add': return theme.colors.node.add;
    case 'Concat2D': return theme.colors.node.concat;
    case 'Output': return theme.colors.node.output;
    default: return theme.colors.text.secondary;
  }
};

export const getNodeLabel = (type: string): string => {
  switch (type) {
    case 'Conv2D': return 'Conv2D';
    case 'Concat2D': return 'Concat';
    default: return type.toString();
  }
};
