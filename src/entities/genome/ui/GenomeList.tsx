import React from 'react';
import { VisualGenome } from '../../../shared/types';
import { saveGenomeToFile } from '../../../shared/api';
import { CheckIcon, SaveIcon, Button } from '../../../shared/ui';
import styles from './GenomeList.module.css';

interface GenomeListProps {
  genomes: VisualGenome[];
}

export const GenomeList: React.FC<GenomeListProps> = ({ genomes }) => {
  if (genomes.length === 0) {
    return (
      <div className={styles.container}>
        <h3 className={styles.title}>Genomes</h3>
        <p className={styles.emptyText}>No genomes available</p>
      </div>
    );
  }

  const handleSave = (genome: VisualGenome) => {
    saveGenomeToFile(genome.genome, () => {
      console.log('Genome saved successfully');
    });
  };

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Genomes</h3>
      <div className={styles.list}>
        {genomes.map((genome) => (
          <div key={genome.id} className={styles.genomeItem}>
            <div className={styles.genomeHeader}>
              <div className={styles.statusContainer}>
                {genome.isValid ? (
                  <div className={styles.validIndicator}>
                    <CheckIcon size={14} />
                    <span>Valid</span>
                  </div>
                ) : (
                  <div className={styles.invalidIndicator}>
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
            <div className={styles.genomeId}>{genome.id}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
