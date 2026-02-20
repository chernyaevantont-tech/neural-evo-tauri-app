import { useCanvasStateStore } from "../../../entities/canvas-state";
import { DeleteConnectionContextMenuItem } from "../../../features/delete-connection"
import { ContextMenu } from "../../../shared/ui/ContextMenu/ContextMenu"

export const ConnectionContextMenu: React.FC = () => {
    const connectionContextMenu = useCanvasStateStore(state => state.connectionContextMenu);
    const setConnectionContextMenu = useCanvasStateStore(state => state.setConnectionContextMenu);

    const cancelContextMenu = () => {
        setConnectionContextMenu(null);
    }

    return (
        <>
            {
                connectionContextMenu &&
                <ContextMenu
                    x={connectionContextMenu.x}
                    y={connectionContextMenu.y}
                >
                    <DeleteConnectionContextMenuItem cancelContextMenu={cancelContextMenu} />
                </ContextMenu>
            }
        </>
    )
}