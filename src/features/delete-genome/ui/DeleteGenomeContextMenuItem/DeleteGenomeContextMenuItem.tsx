import React from "react";
import { useCanvasStateStore } from "../../../../entities/canvas-state";
import { useDeleteGenome } from "../../model/useDeleteGenome";
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu";
import { TrashIcon } from "../../../../shared";

interface DeleteGenomeContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const DeleteGenomeContextMenuItem: React.FC<DeleteGenomeContextMenuItemProps> = ({ cancelContextMenu }) => {
    const contextMenuGenomeId = useCanvasStateStore(state => state.genomeContextMenu?.genomeId);

    const deleteGenome = useDeleteGenome();

    const onClick = () => {
        if (!contextMenuGenomeId) return;
        deleteGenome(contextMenuGenomeId);
    }

    return <ContextMenu.MenuItem icon={<TrashIcon size={14} />} label="Delete Genome" danger cancelContextMenu={cancelContextMenu} onClick={onClick} />
}