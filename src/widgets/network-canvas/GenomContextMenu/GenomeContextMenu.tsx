import { useCanvasStateStore } from "../../../entities/canvas-state";
import { DeleteGenomeContextMenuItem } from "../../../features/delete-genome";
import { TestTrainGenomeContextMenuItem } from "../../../features/train-genome/ui/TestTrainGenomeContextMenuItem";
import { TestTrainOnImageFolderContextMenuItem } from "../../../features/train-genome/ui/TestTrainOnImageFolderContextMenuItem";
import { ContextMenu } from "../../../shared/ui/ContextMenu/ContextMenu"

export const GenomeContextMenu: React.FC = () => {
    const genomeContextMenu = useCanvasStateStore(state => state.genomeContextMenu);
    const setConnectionContextMenu = useCanvasStateStore(state => state.setConnectionContextMenu);

    const cancelContextMenu = () => {
        setConnectionContextMenu(null);
    }

    return (
        <>
            {
                genomeContextMenu &&
                <ContextMenu
                    x={genomeContextMenu.x}
                    y={genomeContextMenu.y}
                >
                    <TestTrainGenomeContextMenuItem cancelContextMenu={cancelContextMenu} />
                    <TestTrainOnImageFolderContextMenuItem cancelContextMenu={cancelContextMenu} />
                    <DeleteGenomeContextMenuItem cancelContextMenu={cancelContextMenu} />
                </ContextMenu>
            }
        </>
    )
}