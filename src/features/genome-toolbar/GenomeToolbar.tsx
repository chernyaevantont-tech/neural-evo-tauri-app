import React from "react";
import styles from './GenomeType.module.css';
import { Button, LoadIcon } from "../../shared/ui";

interface GenomeToolbarProps {
    onLoadGenome: () => void;
    onGetSubgenome: () => void;
}

export const GenomeToolbar: React.FC<GenomeToolbarProps> = ({
    onLoadGenome,
    onGetSubgenome
}) => {
    return (
        <div className={styles.container}>
            <div className={styles.section}>
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
            </div>
        </div>
    )
}