import { useCanvasStateStore } from "../../../entities/canvas-state"

export const useResizeCanvas = () => {
    const scale = useCanvasStateStore(state => state.scale);
    const setScale = useCanvasStateStore(state => state.setScale);
    const translate = useCanvasStateStore(state => state.translate);
    const setTranslate = useCanvasStateStore(state => state.setTranslate);

    return (clientX: number, clientY: number, deltaY: number, rectLeft: number, rectTop: number) => {
        const mouseX = clientX - rectLeft;
        const mouseY = clientY - rectTop;
        const worldX = (mouseX - translate.x) / scale;
        const worldY = (mouseY - translate.y) / scale;

        const delta = deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(5, scale * delta));

        const newTranslateX = mouseX - worldX * newScale;
        const newTranslateY = mouseY - worldY * newScale;

        setScale(newScale);
        setTranslate({ x: newTranslateX, y: newTranslateY });
    }
}