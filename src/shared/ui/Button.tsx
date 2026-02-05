import React, { CSSProperties } from 'react';
import { theme } from '../lib/theme';

interface ButtonProps {
  onClick?: () => void;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  fullWidth?: boolean;
  icon?: React.ReactNode;
  style?: CSSProperties;
}

export const Button: React.FC<ButtonProps> = ({
  onClick,
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  fullWidth = false,
  icon,
  style,
}) => {
  const getVariantStyles = (): CSSProperties => {
    const baseStyle: CSSProperties = {
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.5 : 1,
      transition: theme.transitions.fast,
      border: 'none',
      borderRadius: theme.borderRadius.md,
      fontFamily: theme.typography.fontFamily,
      fontWeight: theme.typography.fontWeight.medium,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: theme.spacing.sm,
      width: fullWidth ? '100%' : 'auto',
    };

    const sizeStyles: Record<string, CSSProperties> = {
      sm: {
        padding: `${theme.spacing.xs} ${theme.spacing.sm}`,
        fontSize: theme.typography.fontSize.sm,
      },
      md: {
        padding: `${theme.spacing.sm} ${theme.spacing.md}`,
        fontSize: theme.typography.fontSize.md,
      },
      lg: {
        padding: `${theme.spacing.md} ${theme.spacing.lg}`,
        fontSize: theme.typography.fontSize.lg,
      },
    };

    const variantStyles: Record<string, CSSProperties> = {
      primary: {
        backgroundColor: theme.colors.accent.primary,
        color: '#ffffff',
      },
      secondary: {
        backgroundColor: theme.colors.background.tertiary,
        color: theme.colors.text.primary,
        border: `1px solid ${theme.colors.border.primary}`,
      },
      danger: {
        backgroundColor: theme.colors.accent.error,
        color: '#ffffff',
      },
      success: {
        backgroundColor: theme.colors.accent.success,
        color: '#ffffff',
      },
    };

    return {
      ...baseStyle,
      ...sizeStyles[size],
      ...variantStyles[variant],
    };
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{ ...getVariantStyles(), ...style }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.filter = 'brightness(1.1)';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.filter = 'brightness(1)';
      }}
    >
      {icon && <span>{icon}</span>}
      {children}
    </button>
  );
};
