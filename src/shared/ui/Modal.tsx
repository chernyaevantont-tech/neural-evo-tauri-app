import React, { CSSProperties, ReactNode } from 'react';
import { theme } from '../lib/theme';
import { CloseIcon } from './Icons';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  maxWidth?: string;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  maxWidth = '600px',
}) => {
  if (!isOpen) return null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div
        style={{ ...modalStyle, maxWidth }}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div style={headerStyle}>
            <h3 style={titleStyle}>{title}</h3>
            <button style={closeButtonStyle} onClick={onClose}>
              <CloseIcon size={18} />
            </button>
          </div>
        )}
        <div style={contentStyle}>{children}</div>
      </div>
    </div>
  );
};

const overlayStyle: CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1000,
  backdropFilter: 'blur(2px)',
};

const modalStyle: CSSProperties = {
  backgroundColor: theme.colors.background.secondary,
  borderRadius: theme.borderRadius.lg,
  boxShadow: theme.shadows.lg,
  maxHeight: '90vh',
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  border: `1px solid ${theme.colors.border.primary}`,
};

const headerStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: theme.spacing.lg,
  borderBottom: `1px solid ${theme.colors.border.primary}`,
};

const titleStyle: CSSProperties = {
  margin: 0,
  fontSize: theme.typography.fontSize.xl,
  fontWeight: theme.typography.fontWeight.semibold,
  color: theme.colors.text.primary,
  fontFamily: theme.typography.fontFamily,
};

const closeButtonStyle: CSSProperties = {
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  padding: theme.spacing.xs,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: theme.colors.text.secondary,
  borderRadius: theme.borderRadius.sm,
  transition: theme.transitions.fast,
};

const contentStyle: CSSProperties = {
  padding: theme.spacing.lg,
  overflow: 'auto',
};
