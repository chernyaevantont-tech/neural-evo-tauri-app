import { useCanvasStateStore } from "../../../entities/canvas-state";
import { DeleteGenomeContextMenuItem } from "../../../features/delete-genome";
import { TestTrainGenomeContextMenuItem } from "../../../features/train-genome/ui/TestTrainGenomeContextMenuItem";
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
                    <DeleteGenomeContextMenuItem cancelContextMenu={cancelContextMenu} />
                </ContextMenu>
            }
        </>
    )
}