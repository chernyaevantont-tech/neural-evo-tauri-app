import { create } from 'zustand';

export type DatasetSourceType = "Folder" | "CSV" | "HuggingFace";

export type DataType =
    | 'Image'
    | 'Vector'
    | 'Categorical'
    | 'Text'
    | 'TemporalSequence';

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

export interface CsvPreprocessingConfig {
    normalization: 'none' | 'global' | 'per-sample' | 'per-channel';
    handleMissing: 'skip' | 'interpolate' | 'mean';
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
    | { type: 'MasterIndex'; indexPath: string; keyField: string; valueField: string; hasHeaders: boolean }
    | { type: 'CsvDataset'; csvPath: string; hasHeaders: boolean; sampleMode: 'row' | 'temporal_window'; featureColumns: string[]; targetColumn: string; windowSize?: number; windowStride?: number; preprocessing: CsvPreprocessingConfig }
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

export const defaultTabularSettings: TabularSettings = {
    normalization: 'min-max',
    oneHot: false,
    fillMissing: 'mean'
};

export const defaultCsvPreprocessing: CsvPreprocessingConfig = {
    normalization: 'per-channel',
    handleMissing: 'skip'
};

export const defaultAugmentation: AugmentationSettings = {
    hFlip: false,
    randomRotation: false,
    randomCrop: false
};

const defaultProfiles: DatasetProfile[] = [];

import { persist, StateStorage, createJSONStorage } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';

// Custom storage engine for Zustand persist using our Tauri commands
const tauriStorage: StateStorage = {
    getItem: async (name: string): Promise<string | null> => {
        try {
            // we only have one file, so we ignore 'name' but we could use it if needed
            const data = await invoke<string>('load_dataset_profiles');
            return data || null;
        } catch (err) {
            console.error('Failed to load dataset profiles:', err);
            return null;
        }
    },
    setItem: async (name: string, value: string): Promise<void> => {
        try {
            await invoke('save_dataset_profiles', { profilesJson: value });
        } catch (err) {
            console.error('Failed to save dataset profiles:', err);
        }
    },
    removeItem: async (name: string): Promise<void> => {
        // Not implemented / needed for this use case
    },
};

export const useDatasetManagerStore = create<DatasetManagerState>()(
    persist(
        (set) => ({
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
        }),
        {
            name: 'dataset-profiles-storage',
            storage: createJSONStorage(() => tauriStorage),
            partialize: (state) => ({ profiles: state.profiles }), // Only persist the profiles array
        }
    )
);
