import React from 'react';
import { Genome } from '../../model/genome';
import styles from './GenomeInfoCard.module.css';

interface GenomeInfoCardProps {
    genome: Genome | null;
}

export const GenomeInfoCard: React.FC<GenomeInfoCardProps> = ({ genome }) => {
    if (!genome) {
        return (
            <div className={styles.container}>
                <h3 className={styles.title}>Genome Information</h3>
                <p className={styles.emptyText}>Select a genome or node to view details</p>
            </div>
        );
    }

    const renderShape = (shape: number[]): string => {
        return `[${shape.join(', ')}]`;
    };

    return (
        <div className={styles.container}>
            <h3 className={styles.title}>Genome Information</h3>

            <div className={styles.section}>
                <div className={styles.label}>Total Nodes</div>
                <div className={styles.value}>{genome.getAllNodes().length}</div>
            </div>

            {genome.inputNodes.length > 0 && (
                <div className={styles.section}>
                    <div className={styles.label}>Input Shapes</div>
                    <div className={styles.value}>
                        <ul className={styles.shapeList}>
                            {genome.inputNodes.map((n, idx) => (
                                <li key={idx} className={styles.inputShapeItem}>
                                    {renderShape(n.GetOutputShape())}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}

            {genome.outputNodes.length > 0 && (
                <div className={styles.section}>
                    <div className={styles.label}>Output Shapes</div>
                    <div className={styles.value}>
                        <ul className={styles.shapeList}>
                            {genome.outputNodes.map((n, idx) => (
                                <li key={idx} className={styles.outputShapeItem}>
                                    {renderShape(n.GetInputShape())}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}

        </div>
    );
};
