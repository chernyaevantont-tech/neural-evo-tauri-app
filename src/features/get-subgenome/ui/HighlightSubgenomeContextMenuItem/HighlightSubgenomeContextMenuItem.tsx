import { LuLightbulb } from "react-icons/lu";
import { useCanvasStateStore } from "../../../../entities/canvas-state"
import { ContextMenu } from "../../../../shared/ui/ContextMenu/ContextMenu";
import { useHighlightSubgenome } from "../../model/useHighlightSubgenome";

export const HighlightSubgenomeContextMenuItem = () => {
    const contextMenuGenomeId = useCanvasStateStore(state => state.genomeContextMenu?.genomeId);

    const hightlightSubgenome = useHighlightSubgenome();

    const onClick = () => {
        if (!contextMenuGenomeId) return;
        hightlightSubgenome(contextMenuGenomeId);
    }

    return <ContextMenu.MenuItem icon={<LuLightbulb size={14}/>} label="Highlight Subgenome" onClick={onClick}/>
}