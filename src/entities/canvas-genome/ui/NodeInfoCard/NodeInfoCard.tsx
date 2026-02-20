import React from 'react';
import { VisualNode } from '../../model/types';
import styles from './NodeInfoCard.module.css';

interface NodeInfoCardProps {
  node: VisualNode | null;
}

export const NodeInfoCard: React.FC<NodeInfoCardProps> = ({ node }) => {
  if (!node) {
    return (
      <div className={styles.container}>
        <h3 className={styles.title}>Node Information</h3>
        <p className={styles.emptyText}>Select a node to view details</p>
      </div>
    );
  }

  let info: any;
  try {
    info = JSON.parse(node.node.GetInfo() as string);
  } catch (e) {
    info = { error: 'Failed to parse node info' };
  }

  const renderValue = (value: any): string => {
    if (Array.isArray(value)) {
      return `[${value.join(', ')}]`;
    }
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Node Information</h3>

      <div className={styles.section}>
        <div className={styles.label}>Type</div>
        <div className={styles.value}>{info.node}</div>
      </div>

      {info.params && Object.keys(info.params).length > 0 && (
        <div className={styles.section}>
          <div className={styles.label}>Parameters</div>
          <div className={styles.paramsContainer}>
            {Object.entries(info.params).map(([key, value]) => (
              <div key={key} className={styles.paramRow}>
                <span className={styles.paramKey}>{key}</span>
                <span className={styles.paramValue}>{renderValue(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.section}>
        <div className={styles.label}>Output Shape</div>
        <div className={styles.value}>{renderValue(node.node.GetOutputShape())}</div>
      </div>

      <div className={styles.section}>
        <div className={styles.label}>Node ID</div>
        <div className={styles.idStyle}>{node.node.id}</div>
      </div>
    </div>
  );
};
