import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface CanvasState {
    selectedNodeId: string | null;
    selectedGenomeId: string | null;
    selectedConnectionId: string | null;
    draggingNodeId: string | null;
    connectingFromNodeId: string | null;
    breedingStartGenomeId: string | null;
    dragOffset: { x: number, y: number };
    scale: number;
    translate: { x: number, y: number };
    isPanning: boolean;
    lastPanningPos: { x: number, y: number };
    nodeContextMenu: { x: number, y: number, nodeId: string } | null;
    genomeContextMenu: { x: number, y: number, genomeId: string } | null;
    connectionContextMenu: { x: number, y: number, connectionId: string } | null;
    canvasWidth: number;
    canvasHeight: number;
    setSelectedNodeId: (nodeId: string | null) => void;
    setSelectedGenomeId: (genomeId: string | null) => void;
    setSelectedConnectionId: (connectionId: string | null) => void;
    setDraggingNodeId: (nodeId: string | null) => void;
    setConnectingFromNodeId: (nodeId: string | null) => void;
    setBreedingStartGenomeId: (genomeId: string | null) => void;
    setDragOffset: (position: { x: number, y: number }) => void;
    setScale: (scale: number) => void;
    setTranslate: (translate: { x: number, y: number }) => void;
    setIsPanning: (isPanning: boolean) => void;
    setLastPanningPos: (position: { x: number, y: number }) => void;
    setZoom: (mouseX: number, mouseY: number, deltaY: number) => void;
    setNodeContextMenu: (contextMenu: { x: number, y: number, nodeId: string } | null) => void;
    setGenomeContextMenu: (contextMenu: { x: number, y: number, genomeId: string } | null) => void;
    setConnectionContextMenu: (contextMenu: { x: number, y: number, connectionId: string } | null) => void;
    setCanvasWidth: (width: number) => void;
    setCanvasHeight: (height: number) => void;
    reset: () => void;
}

export const useCanvasStateStore = create<CanvasState>()(
    immer((set) => ({
        selectedNodeId: null,
        selectedGenomeId: null,
        selectedConnectionId: null,
        draggingNodeId: null,
        connectingFromNodeId: null,
        breedingStartGenomeId: null,
        dragOffset: { x: 0, y: 0 },
        scale: 1,
        translate: { x: 0, y: 0 },
        isPanning: false,
        lastPanningPos: { x: 0, y: 0 },
        nodeContextMenu: null,
        genomeContextMenu: null,
        connectionContextMenu: null,
        canvasWidth: 0,
        canvasHeight: 0,
        setSelectedNodeId: (nodeId) =>
            set(state => {
                state.selectedNodeId = nodeId;
                state.selectedGenomeId = null;
                state.selectedConnectionId = null;
            }),
        setSelectedGenomeId: (genomeId) =>
            set(state => {
                state.selectedNodeId = null;
                state.selectedGenomeId = genomeId;
                state.selectedConnectionId = null;
            }),
        setSelectedConnectionId: (connectionId) =>
            set(state => {
                state.selectedNodeId = null;
                state.selectedGenomeId = null;
                state.selectedConnectionId = connectionId;
            }),
        setDraggingNodeId: (nodeId) => set(state => { state.draggingNodeId = nodeId }),
        setConnectingFromNodeId: (nodeId) => set(state => { state.connectingFromNodeId = nodeId }),
        setBreedingStartGenomeId: (genomeId) => set(state => { state.breedingStartGenomeId = genomeId }),
        setDragOffset: (position) => set(state => { state.dragOffset = position }),
        setScale: (scale) => set(state => { state.scale = scale }),
        setTranslate: (translate) => set(state => { state.translate = translate }),
        setIsPanning: (isPanning) => set(state => { state.isPanning = isPanning }),
        setLastPanningPos: (position) => set(state => { state.lastPanningPos = position }),
        setZoom: (mouseX, mouseY, deltaY) =>
            set(state => {
                const worldX = (mouseX - state.translate.x);
                const worldY = (mouseY - state.translate.y);

                const delta = deltaY > 0 ? 0.9 : 1.1;
                const newScale = Math.max(0.1, Math.min(5, state.scale * delta));

                const newTranslateX = mouseX - worldX * newScale;
                const newTranslateY = mouseY - worldY * newScale;

                state.scale = newScale;
                state.translate = { x: newTranslateX, y: newTranslateY };
            }),
        setNodeContextMenu: (contextMenu) =>
            set(state => {
                state.nodeContextMenu = contextMenu;
                state.genomeContextMenu = null;
                state.connectionContextMenu = null;
            }),
        setGenomeContextMenu: (contextMenu) =>
            set(state => {
                state.nodeContextMenu = null;
                state.genomeContextMenu = contextMenu;
                state.connectionContextMenu = null;
            }),
        setConnectionContextMenu: (contextMenu) =>
            set(state => {
                state.nodeContextMenu = null;
                state.genomeContextMenu = null;
                state.connectionContextMenu = contextMenu;
            }),
        setCanvasWidth: (width) => set(state => { state.canvasWidth = width }),
        setCanvasHeight: (height) => set(state => { state.canvasHeight = height }),
        reset: () => set(state => {
            state.selectedNodeId = null;
            state.selectedGenomeId = null;
            state.selectedConnectionId = null;
            state.draggingNodeId = null;
            state.connectingFromNodeId = null;
            state.breedingStartGenomeId = null;
            state.dragOffset = { x: 0, y: 0 };
            state.scale = 1;
            state.translate = { x: 0, y: 0 };
            state.isPanning = false;
            state.lastPanningPos = { x: 0, y: 0 };
            state.nodeContextMenu = null;
            state.genomeContextMenu = null;
            state.connectionContextMenu = null;
        }),
    }))
)