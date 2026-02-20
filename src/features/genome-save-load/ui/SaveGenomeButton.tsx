import { Genome, VisualGenome } from "../../../entities/canvas-genome"
import { Button, SaveIcon } from "../../../shared"
import { useSaveGenome } from "../model/useSaveGenome"

interface SaveGenomeButtonProps {
    genome: Genome
}

export const SaveGenomeButton: React.FC<SaveGenomeButtonProps> = ({genome}) => {
    const saveGenome = useSaveGenome();
    
    return (
        <Button
            variant="secondary"
            size="sm"
            icon={<SaveIcon size={14} />}
            onClick={() => saveGenome(genome)}
        >
            Save
        </Button>
    )
}