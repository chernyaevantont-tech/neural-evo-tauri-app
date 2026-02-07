import React from 'react';
import { Button, LoadIcon, PlusIcon } from '../../shared/ui';
import styles from './NodeToolbar.module.css';

interface NodeToolbarProps {
  onAddNode: (nodeType: string) => void;
  onLoadGenome: () => void;
  onGetSubgenome: () => void;
}

export const NodeToolbar: React.FC<NodeToolbarProps> = ({
  onAddNode,
  onLoadGenome,
  onGetSubgenome,
}) => {
  const nodeTypes = [
    { type: 'Input', label: 'Input' },
    { type: 'Dense', label: 'Dense' },
    { type: 'Conv2D', label: 'Conv2D' },
    { type: 'Pooling', label: 'Pooling' },
    { type: 'Flatten', label: 'Flatten' },
    { type: 'Add', label: 'Add' },
    { type: 'Concat2D', label: 'Concat' },
    { type: 'Output', label: 'Output' },
  ];

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <h4 className={styles.sectionTitle}>Add Layers</h4>
        <div className={styles.buttonGrid}>
          {nodeTypes.map(({ type, label }) => (
            <Button
              key={type}
              onClick={() => onAddNode(type)}
              variant="secondary"
              size="sm"
              icon={<PlusIcon size={14} />}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      <div className={styles.divider} />

      {/* <div className={styles.section}>
        <h4 className={styles.sectionTitle}>Genome Operations</h4>
        <div className={styles.operations}>
          <Button
            onClick={onLoadGenome}
            variant="primary"
            size="md"
            icon={<LoadIcon size={16} />}
            fullWidth
          >
            Load Genome
          </Button>
          <Button
            onClick={onGetSubgenome}
            variant="secondary"
            size="md"
            fullWidth
          >
            Get Subgenome
          </Button>
        </div>
      </div> */}
    </div>
  );
};
