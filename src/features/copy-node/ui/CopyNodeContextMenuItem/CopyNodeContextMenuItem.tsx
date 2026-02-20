import { useCanvasStateStore } from "../../../../entities/canvas-state"
import { CopyIcon } from "../../../../shared";
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu";
import { useCopyNode } from "../../model/useCopyNode";

interface CopyNodeContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const CopyNodeContextMenuItem: React.FC<CopyNodeContextMenuItemProps> = ({cancelContextMenu}) => {
    const contextMenuNodeId = useCanvasStateStore(state => state.nodeContextMenu?.nodeId);
    
    const copyNode = useCopyNode();

    const onClick = () => {
        if (!contextMenuNodeId) return;
        copyNode(contextMenuNodeId)
    }

    return <ContextMenu.MenuItem icon={<CopyIcon size={14}/>} label="Copy Node" cancelContextMenu={cancelContextMenu} onClick={onClick}/>
}