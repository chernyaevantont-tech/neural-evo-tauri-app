import React, { CSSProperties } from 'react';
import { theme } from '../../shared/lib';
import { EditIcon, TrashIcon, CopyIcon } from '../../shared/ui';

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
      style={{
        ...menuItemStyle,
        color: danger ? theme.colors.accent.error : theme.colors.text.primary,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = theme.colors.background.hover;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent';
      }}
    >
      <span style={iconStyle}>{icon}</span>
      {label}
    </button>
  );

  return (
    <div
      style={{ ...containerStyle, left: x, top: y }}
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

const containerStyle: CSSProperties = {
  position: 'fixed',
  backgroundColor: theme.colors.background.secondary,
  border: `1px solid ${theme.colors.border.primary}`,
  borderRadius: theme.borderRadius.md,
  boxShadow: theme.shadows.lg,
  zIndex: 2000,
  minWidth: '160px',
  overflow: 'hidden',
};

const menuItemStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.sm,
  width: '100%',
  padding: `${theme.spacing.sm} ${theme.spacing.md}`,
  fontSize: theme.typography.fontSize.md,
  fontFamily: theme.typography.fontFamily,
  border: 'none',
  background: 'transparent',
  cursor: 'pointer',
  textAlign: 'left',
  transition: theme.transitions.fast,
};

const iconStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};
