import { EditIcon } from "../../../../shared"
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu"


interface EditNodeContextMenuItemProps {
    setIsModalOpen: (isModalOpen: boolean) => void
}

export const EditNodeContextMenuItem: React.FC<EditNodeContextMenuItemProps> = ({setIsModalOpen}) => {
    const onClick = () => {
        setIsModalOpen(true);
    }

    return  <ContextMenu.MenuItem icon={<EditIcon size={14}/>} label="Edit Node" onClick={onClick}/>
}