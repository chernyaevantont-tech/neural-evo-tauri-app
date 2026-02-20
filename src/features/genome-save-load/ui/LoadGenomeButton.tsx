import { BaseNode, Genome } from "../../../entities/canvas-genome";
import { Button, LoadIcon } from "../../../shared"
import { useLoadGenome } from "../model/useLoadGenome";

export const LoadGenomeButton = () => {
    const loadGenome = useLoadGenome();
    
    return (
        <Button
            onClick={async () => {
                await loadGenome()
            }}
            variant="primary"
            size="md"
            icon={<LoadIcon size={16} />}
            fullWidth
        >
            Load Genome
        </Button>
    )
}