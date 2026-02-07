import React from 'react';
import { EditIcon, TrashIcon, CopyIcon } from '../../shared/ui';
import styles from './ContextMenu.module.css';

interface ContextMenuProps {
  x: number;
  y: number;
  onEdit?: () => void;
  onCopy?: () => void;
  onDelete: () => void;
  type: 'node' | 'connection';
}

export const ContextMenu: React.FC<ContextMenuProps> = ({
  x,
  y,
  onEdit,
  onCopy,
  onDelete,
  type,
}) => {
  const MenuItem: React.FC<{
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
    danger?: boolean;
  }> = ({ icon, label, onClick, danger }) => (
    <button
      onClick={onClick}
      className={`${styles.menuItem} ${danger ? styles.danger : ''}`}
    >
      <span className={styles.icon}>{icon}</span>
      {label}
    </button>
  );

  return (
    <div
      className={styles.container}
      style={{ left: x, top: y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {type === 'node' && (
        <>
          {onEdit && (
            <MenuItem icon={<EditIcon size={14} />} label="Edit Node" onClick={onEdit} />
          )}
          {onCopy && (
            <MenuItem icon={<CopyIcon size={14} />} label="Copy Node" onClick={onCopy} />
          )}
        </>
      )}
      <MenuItem
        icon={<TrashIcon size={14} />}
        label={`Delete ${type === 'node' ? 'Node' : 'Connection'}`}
        onClick={onDelete}
        danger
      />
    </div>
  );
};
