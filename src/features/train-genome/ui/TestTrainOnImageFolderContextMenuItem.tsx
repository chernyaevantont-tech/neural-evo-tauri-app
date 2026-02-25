import { CgImage } from "react-icons/cg";
import { useCanvasStateStore } from "../../../entities/canvas-state";
import { ContextMenu } from "../../../shared/ui/ContextMenu/ContextMenu";
import { useTestTrainOnImageFolder } from "../model/useTestTrainGenome";

interface TestTrainOnImageFolderContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const TestTrainOnImageFolderContextMenuItem: React.FC<TestTrainOnImageFolderContextMenuItemProps> = ({ cancelContextMenu }) => {
    const contextMenuGenomeId = useCanvasStateStore(state => state.genomeContextMenu?.genomeId);

    const testTrainOnImageFolder = useTestTrainOnImageFolder();

    const onClick = () => {
        if (!contextMenuGenomeId) return;
        testTrainOnImageFolder(contextMenuGenomeId);
    }

    return <ContextMenu.MenuItem icon={<CgImage size={14} />} label="Train on Local Images" cancelContextMenu={cancelContextMenu} onClick={onClick} />
}
