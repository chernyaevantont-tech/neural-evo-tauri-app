import React from "react";
import { useCanvasStateStore } from "../../../../entities/canvas-state";
import { useDeleteNode } from "../../model/useDeleteNode";
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu";
import { TrashIcon } from "../../../../shared";

interface DeleteNodeContextMenuItem {
    cancelContextMenu: () => void;
}

export const DeleteNodeContextMenuItem: React.FC<DeleteNodeContextMenuItem> = ({cancelContextMenu}) => {
    const contextMenuNodeId = useCanvasStateStore(state => state.nodeContextMenu?.nodeId);

    const deleteNode = useDeleteNode();

    const onClick = () => {
        if (!contextMenuNodeId) return;
        deleteNode(contextMenuNodeId);
    }

    return <ContextMenu.MenuItem icon={<TrashIcon size={14}/>} label="Delete Node" danger cancelContextMenu={cancelContextMenu} onClick={onClick}/>
}