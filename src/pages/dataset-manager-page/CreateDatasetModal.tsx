import React, { useState } from 'react';
import styles from './CreateDatasetModal.module.css';
import { BsFolder, BsXLg } from 'react-icons/bs';
import { DatasetSourceType, DatasetProfile, useDatasetManagerStore, defaultAugmentation } from '../../features/dataset-manager/model/store';
import { invoke } from '@tauri-apps/api/core';

interface Props {
    onClose: () => void;
}

export const CreateDatasetModal: React.FC<Props> = ({ onClose }) => {
    const addProfile = useDatasetManagerStore(s => s.addProfile);
    const [name, setName] = useState('');
    const [sourcePath, setSourcePath] = useState('');

    const handleSelectFolder = async () => {
        try {
            const selected = await invoke<string>("pick_folder");
            if (selected && selected.length > 0) {
                setSourcePath(selected);
                if (!name) {
                    const parts = selected.split(/[\\/]/);
                    setName(parts[parts.length - 1]);
                }
            }
        } catch (e) {
            console.error("Failed to open dialog via Rust", e);
        }
    };

    const handleCreate = () => {
        if (!name || !sourcePath) return;

        const baseProfile: Partial<DatasetProfile> = {
            id: crypto.randomUUID(),
            name,
            type: 'Folder',
            sourcePath,
            streams: [],
            split: { train: 80, val: 10, test: 10 },
            augmentation: { ...defaultAugmentation },
            isScanned: false
        };

        addProfile(baseProfile as DatasetProfile);
        onClose();
    };

    return (
        <div className={styles.overlay}>
            <div className={styles.modal}>
                <div className={styles.header}>
                    <h2>Create Dataset Profile</h2>
                    <BsXLg className={styles.closeIcon} onClick={onClose} />
                </div>

                <div className={styles.body}>
                    <div className={styles.inputGroup}>
                        <label>Profile Name</label>
                        <input
                            type="text"
                            placeholder="e.g. My Custom Cats/Dogs"
                            value={name}
                            onChange={e => setName(e.target.value)}
                        />
                    </div>

                    <div className={styles.inputGroup}>
                        <label>Directory Path</label>
                        <div className={styles.folderRow}>
                            <input
                                type="text"
                                readOnly
                                value={sourcePath}
                                placeholder="Select a folder..."
                                style={{ flex: 1 }}
                            />
                            <button className={styles.browseBtn} onClick={handleSelectFolder}>
                                <BsFolder /> Browse
                            </button>
                        </div>
                        <small className={styles.hint}>
                            Select the root folder of your project. You can define specific data streams (images, files) later.
                        </small>
                    </div>
                </div>

                <div className={styles.footer}>
                    <button className={styles.cancelBtn} onClick={onClose}>Cancel</button>
                    <button
                        className={styles.createBtn}
                        onClick={handleCreate}
                        disabled={!name || !sourcePath}
                    >
                        Create
                    </button>
                </div>
            </div>
        </div>
    );
}
