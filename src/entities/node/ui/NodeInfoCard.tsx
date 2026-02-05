import React, { CSSProperties } from 'react';
import { VisualNode } from '../../../shared/types';
import { theme } from '../../../shared/lib';

interface NodeInfoCardProps {
  node: VisualNode | null;
}

export const NodeInfoCard: React.FC<NodeInfoCardProps> = ({ node }) => {
  if (!node) {
    return (
      <div style={containerStyle}>
        <h3 style={titleStyle}>Node Information</h3>
        <p style={emptyTextStyle}>Select a node to view details</p>
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
    <div style={containerStyle}>
      <h3 style={titleStyle}>Node Information</h3>

      <div style={sectionStyle}>
        <div style={labelStyle}>Type</div>
        <div style={valueStyle}>{info.node}</div>
      </div>

      {info.params && Object.keys(info.params).length > 0 && (
        <div style={sectionStyle}>
          <div style={labelStyle}>Parameters</div>
          <div style={paramsContainer}>
            {Object.entries(info.params).map(([key, value]) => (
              <div key={key} style={paramRowStyle}>
                <span style={paramKeyStyle}>{key}</span>
                <span style={paramValueStyle}>{renderValue(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={sectionStyle}>
        <div style={labelStyle}>Output Shape</div>
        <div style={valueStyle}>{renderValue(node.node.GetOutputShape())}</div>
      </div>

      <div style={sectionStyle}>
        <div style={labelStyle}>Node ID</div>
        <div style={idStyle}>{node.node.id}</div>
      </div>
    </div>
  );
};

const containerStyle: CSSProperties = {
  backgroundColor: theme.colors.background.secondary,
  padding: theme.spacing.lg,
  borderRadius: theme.borderRadius.lg,
  border: `1px solid ${theme.colors.border.primary}`,
};

const titleStyle: CSSProperties = {
  margin: `0 0 ${theme.spacing.lg} 0`,
  fontSize: theme.typography.fontSize.xl,
  fontWeight: theme.typography.fontWeight.semibold,
  color: theme.colors.text.primary,
  fontFamily: theme.typography.fontFamily,
  borderBottom: `2px solid ${theme.colors.accent.primary}`,
  paddingBottom: theme.spacing.sm,
};

const emptyTextStyle: CSSProperties = {
  color: theme.colors.text.secondary,
  fontStyle: 'italic',
  textAlign: 'center',
  padding: theme.spacing.xl,
  fontFamily: theme.typography.fontFamily,
  fontSize: theme.typography.fontSize.md,
};

const sectionStyle: CSSProperties = {
  marginBottom: theme.spacing.md,
};

const labelStyle: CSSProperties = {
  fontSize: theme.typography.fontSize.xs,
  fontWeight: theme.typography.fontWeight.semibold,
  color: theme.colors.text.secondary,
  textTransform: 'uppercase',
  marginBottom: theme.spacing.xs,
  fontFamily: theme.typography.fontFamily,
  letterSpacing: '0.5px',
};

const valueStyle: CSSProperties = {
  fontSize: theme.typography.fontSize.md,
  color: theme.colors.text.primary,
  padding: theme.spacing.sm,
  backgroundColor: theme.colors.background.tertiary,
  borderRadius: theme.borderRadius.md,
  fontFamily: theme.typography.fontFamily,
  border: `1px solid ${theme.colors.border.primary}`,
};

const paramsContainer: CSSProperties = {
  backgroundColor: theme.colors.background.tertiary,
  borderRadius: theme.borderRadius.md,
  padding: theme.spacing.sm,
  border: `1px solid ${theme.colors.border.primary}`,
};

const paramRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  padding: `${theme.spacing.xs} 0`,
  borderBottom: `1px solid ${theme.colors.border.primary}`,
  fontFamily: theme.typography.fontFamily,
};

const paramKeyStyle: CSSProperties = {
  fontWeight: theme.typography.fontWeight.medium,
  color: theme.colors.text.primary,
  fontSize: theme.typography.fontSize.sm,
};

const paramValueStyle: CSSProperties = {
  color: theme.colors.text.accent,
  fontSize: theme.typography.fontSize.sm,
  textAlign: 'right',
  maxWidth: '60%',
  wordBreak: 'break-word',
};

const idStyle: CSSProperties = {
  ...valueStyle,
  fontSize: theme.typography.fontSize.xs,
  color: theme.colors.text.tertiary,
  fontFamily: 'monospace',
};
