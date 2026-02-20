import React from "react";
import { useCanvasStateStore } from "../../../../entities/canvas-state";
import { useDeleteConnection } from "../../model/useDeleteConnection";
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu";
import { TrashIcon } from "../../../../shared";

interface DeleteConnectionContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const DeleteConnectionContextMenuItem: React.FC<DeleteConnectionContextMenuItemProps> = ({cancelContextMenu}) => {
    const contextMenuNodeId = useCanvasStateStore(state => state.connectionContextMenu?.connectionId);

    const deleteConnection = useDeleteConnection();

    const onClick = () => {
        if (!contextMenuNodeId) return;
        deleteConnection(contextMenuNodeId);
    }

    return <ContextMenu.MenuItem icon={<TrashIcon size={14}/>} label="Delete Connection" danger cancelContextMenu={cancelContextMenu} onClick={onClick}/>
}