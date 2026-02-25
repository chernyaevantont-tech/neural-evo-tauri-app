import { CgDanger } from "react-icons/cg";
import { useCanvasStateStore } from "../../../entities/canvas-state";
import { ContextMenu } from "../../../shared/ui/ContextMenu/ContextMenu";
import { useTestTrainGenome } from "../model/useTestTrainGenome";

interface TestTrainGenomeContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const TestTrainGenomeContextMenuItem: React.FC<TestTrainGenomeContextMenuItemProps> = ({ cancelContextMenu }) => {
    const contextMenuGenomeId = useCanvasStateStore(state => state.genomeContextMenu?.genomeId);

    const testTrainGenome = useTestTrainGenome();

    const onClick = () => {
        if (!contextMenuGenomeId) return;
        testTrainGenome(contextMenuGenomeId);
    }

    return <ContextMenu.MenuItem icon={<CgDanger size={14} />} label="Test Train Genome" cancelContextMenu={cancelContextMenu} onClick={onClick} />
} 