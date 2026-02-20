import { useState } from "react";
import { CopyNodeContextMenuItem } from "../../../features/copy-node";
import { DeleteNodeContextMenuItem } from "../../../features/delete-node";
import { EditNodeContextMenuItem, useEditNode } from "../../../features/edit-node";
import { ContextMenu } from "../../../shared/ui/ContextMenu/ContextMenu"
import { EditNodeModal } from "../../../features/edit-node/ui/EditNodeModal/EditNodeModal";
import { useCanvasStateStore } from "../../../entities/canvas-state";
import { BaseNode, useCanvasGenomeStore } from "../../../entities/canvas-genome";

export const NodeContextMenu: React.FC = () => {
    const contextMenuNodeId = useCanvasStateStore(state => state.nodeContextMenu?.nodeId);
    const nodeContextMenu = useCanvasStateStore(state => state.nodeContextMenu);
    const setNodeContextMenu = useCanvasStateStore(state => state.setNodeContextMenu);
    const node = useCanvasGenomeStore(state => contextMenuNodeId ? state.nodes.get(contextMenuNodeId) : null);

    const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

    const editNode = useEditNode();

    const onSave = (node: BaseNode) => {
        if (!contextMenuNodeId) return;
        setIsModalOpen(false);
        editNode(contextMenuNodeId, node);
    }

    const onCancel = () => {
        setIsModalOpen(false);
    }

    const cancelContextMenu = () => {
        setNodeContextMenu(null);
    }

    return (
        <>
            {!isModalOpen &&nodeContextMenu && <ContextMenu
                x={nodeContextMenu.x}
                y={nodeContextMenu.y}
            >
                <CopyNodeContextMenuItem cancelContextMenu={cancelContextMenu} />
                <EditNodeContextMenuItem setIsModalOpen={setIsModalOpen} />
                <DeleteNodeContextMenuItem cancelContextMenu={cancelContextMenu} />
            </ContextMenu>}
            {isModalOpen && node && <EditNodeModal
                nodeType={node.node.GetNodeType()}
                existingNode={node.node}
                onSave={(node) => { onSave(node); cancelContextMenu(); }}
                onCancel={() => { onCancel(); cancelContextMenu(); }}
            />}
        </>
    )
}