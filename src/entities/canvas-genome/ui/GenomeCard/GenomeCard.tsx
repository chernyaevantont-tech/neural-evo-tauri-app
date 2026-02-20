import { ReactNode } from "react"
import { CheckIcon } from "../../../../shared"
import styles from "./GenomeCard.module.css"

interface GenomeCardProps {
    genomeId: string,
    isValid: boolean,
    actionSlot?: ReactNode,
}

export const GenomeCard: React.FC<GenomeCardProps> = ({
    genomeId,
    isValid,
    actionSlot
}) => {
    return (
        <div className={styles.genomeItem}>
            <div className={styles.genomeHeader}>
                <div className={styles.statusContainer}>
                    {isValid ? (
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
                {actionSlot}
            </div>
            <div className={styles.genomeId}>{genomeId}</div>
        </div>
    )
}