import { create } from 'zustand';

export type DatasetSourceType = "Folder" | "CSV" | "HuggingFace";

export type DataType =
    | 'Image'
    | 'Vector'
    | 'Categorical'
    | 'Text';

export interface VisionSettings {
    resize: [number, number];
    grayscale: boolean;
    normalization: '0-1' | 'imagenet' | 'none';
}

export interface TabularSettings {
    normalization: 'min-max' | 'z-score' | 'none';
    oneHot: boolean;
    fillMissing: 'mean' | 'median' | 'mode' | 'drop';
}

export interface AugmentationSettings {
    hFlip: boolean;
    randomRotation: boolean;
    randomCrop: boolean;
}

export type DataLocatorDef =
    | { type: 'GlobPattern'; pattern: string }
    | { type: 'FolderMapping' } // Uses parent folder name
    | { type: 'CompanionFile'; pathTemplate: string; parser: 'YOLO' | 'Text' | 'COCO_Subset' }
    | { type: 'MasterIndex'; indexPath: string; keyField: string; valueField: string }
    | { type: 'None' }; // Fallback

export interface DataStream {
    id: string;
    alias: string;        // E.g., "Image Input", "Price Output"
    role: 'Input' | 'Target' | 'Ignore';
    dataType: DataType;
    tensorShape: number[]; // Explicit shape expected by the model

    // How the data is found and aligned by SampleID
    locator: DataLocatorDef;

    preprocessing?: {
        vision?: VisionSettings;
        tabular?: TabularSettings;
    }
}

export interface StreamScanInfo {
    streamId: string;
    alias: string;
    foundCount: number;       // How many SampleIDs this stream resolved
    missingSampleIds: string[]; // SampleIDs that could not be resolved for THIS stream
    discoveredClasses?: Record<string, number>; // class_name -> count (for FolderMapping)
}

export interface ScanResult {
    totalMatched: number;        // Final sample count after intersection
    droppedCount: number;        // Samples dropped due to missing data in at least one stream
    streamReports: StreamScanInfo[];
    timestamp: string;           // ISO string of when the scan happened
}

export interface DatasetProfile {
    id: string;
    name: string;
    type: DatasetSourceType;
    sourcePath?: string; // Path or URL

    // Data streams definitions
    streams: DataStream[];

    totalSamples?: number;
    split: { train: number, val: number, test: number };

    // Global Augmentations (applied mostly to vision streams if present)
    augmentation: AugmentationSettings;

    // Scan & Validation
    scanResult?: ScanResult;
    isScanned: boolean;
}

interface DatasetManagerState {
    profiles: DatasetProfile[];
    selectedProfileId: string | null;

    // Actions
    addProfile: (profile: DatasetProfile) => void;
    updateProfile: (id: string, updates: Partial<DatasetProfile>) => void;
    removeProfile: (id: string) => void;
    setSelectedProfileId: (id: string | null) => void;
}

export const defaultVisionSettings: VisionSettings = {
    resize: [64, 64],
    grayscale: false,
    normalization: '0-1'
};

export const defaultAugmentation: AugmentationSettings = {
    hFlip: false,
    randomRotation: false,
    randomCrop: false
};

const defaultProfiles: DatasetProfile[] = [];

export const useDatasetManagerStore = create<DatasetManagerState>((set) => ({
    profiles: defaultProfiles,
    selectedProfileId: null,

    addProfile: (profile) => set((state) => ({
        profiles: [...state.profiles, profile],
        selectedProfileId: profile.id
    })),

    updateProfile: (id, updates) => set((state) => ({
        profiles: state.profiles.map(p => p.id === id ? { ...p, ...updates } : p)
    })),

    removeProfile: (id) => set((state) => ({
        profiles: state.profiles.filter(p => p.id !== id),
        selectedProfileId: state.selectedProfileId === id ? null : state.selectedProfileId
    })),

    setSelectedProfileId: (id) => set({ selectedProfileId: id })
}));
