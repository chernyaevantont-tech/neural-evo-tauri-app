import React, { CSSProperties } from 'react';
import { VisualGenome } from '../../../shared/types';
import { theme } from '../../../shared/lib';
import { saveGenomeToFile } from '../../../shared/api';
import { CheckIcon, SaveIcon, Button } from '../../../shared/ui';

interface GenomeListProps {
  genomes: VisualGenome[];
}

export const GenomeList: React.FC<GenomeListProps> = ({ genomes }) => {
  if (genomes.length === 0) {
    return (
      <div style={containerStyle}>
        <h3 style={titleStyle}>Genomes</h3>
        <p style={emptyTextStyle}>No genomes available</p>
      </div>
    );
  }

  const handleSave = (genome: VisualGenome) => {
    saveGenomeToFile(genome.genome, () => {
      console.log('Genome saved successfully');
    });
  };

  return (
    <div style={containerStyle}>
      <h3 style={titleStyle}>Genomes</h3>
      <div style={listStyle}>
        {genomes.map((genome) => (
          <div key={genome.id} style={genomeItemStyle}>
            <div style={genomeHeaderStyle}>
              <div style={statusContainerStyle}>
                {genome.isValid ? (
                  <div style={validIndicatorStyle}>
                    <CheckIcon size={14} />
                    <span>Valid</span>
                  </div>
                ) : (
                  <div style={invalidIndicatorStyle}>
                    <span>Invalid</span>
                  </div>
                )}
              </div>
              <Button
                variant="secondary"
                size="sm"
                icon={<SaveIcon size={14} />}
                onClick={() => handleSave(genome)}
              >
                Save
              </Button>
            </div>
            <div style={genomeIdStyle}>{genome.id}</div>
          </div>
        ))}
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

const listStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing.sm,
};

const genomeItemStyle: CSSProperties = {
  backgroundColor: theme.colors.background.tertiary,
  padding: theme.spacing.md,
  borderRadius: theme.borderRadius.md,
  border: `1px solid ${theme.colors.border.primary}`,
};

const genomeHeaderStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing.sm,
};

const statusContainerStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
};

const validIndicatorStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
  color: theme.colors.accent.success,
  fontSize: theme.typography.fontSize.sm,
  fontWeight: theme.typography.fontWeight.medium,
  fontFamily: theme.typography.fontFamily,
};

const invalidIndicatorStyle: CSSProperties = {
  color: theme.colors.accent.error,
  fontSize: theme.typography.fontSize.sm,
  fontWeight: theme.typography.fontWeight.medium,
  fontFamily: theme.typography.fontFamily,
};

const genomeIdStyle: CSSProperties = {
  fontSize: theme.typography.fontSize.xs,
  color: theme.colors.text.tertiary,
  fontFamily: 'monospace',
  wordBreak: 'break-all',
};
