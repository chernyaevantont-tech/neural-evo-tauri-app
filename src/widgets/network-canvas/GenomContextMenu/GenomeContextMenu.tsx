import { useCanvasStateStore } from "../../../entities/canvas-state";
import { DeleteGenomeContextMenuItem } from "../../../features/delete-genome";
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
                    <DeleteGenomeContextMenuItem cancelContextMenu={cancelContextMenu} />
                </ContextMenu>
            }
        </>
    )
}