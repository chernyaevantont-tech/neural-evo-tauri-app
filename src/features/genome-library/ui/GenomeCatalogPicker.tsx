import React, { useEffect, useMemo, useState } from 'react';
import styles from './GenomeCatalogPicker.module.css';
import { useGenomeLibraryStore, GenomeLibraryEntry } from '../model/store';

interface GenomeCatalogPickerProps {
    /** Title shown in the modal header */
    title?: string;
    /** Whether to allow selecting multiple genomes */
    multi?: boolean;
    /** Called when a genome is selected (single mode) */
    onSelect?: (entry: GenomeLibraryEntry) => void;
    /** Called when genomes are confirmed (multi mode) */
    onConfirm?: (entries: GenomeLibraryEntry[]) => void;
    /** Called when the picker is closed */
    onClose: () => void;
}

/**
 * Reusable modal for browsing and selecting genomes from the library.
 * Supports single-select (click to pick) and multi-select (checkboxes + confirm).
 */
export const GenomeCatalogPicker: React.FC<GenomeCatalogPickerProps> = ({
    title = 'Select Genome',
    multi = false,
    onSelect,
    onConfirm,
    onClose,
}) => {
    const { entries, loadLibrary } = useGenomeLibraryStore();
    const [search, setSearch] = useState('');
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

    useEffect(() => {
        loadLibrary();
    }, [loadLibrary]);

    const filtered = useMemo(() => {
        const q = search.toLowerCase();
        return entries.filter(e =>
            e.name.toLowerCase().includes(q) ||
            e.tags.some(t => t.toLowerCase().includes(q)) ||
            e.layerTypes.some(l => l.toLowerCase().includes(q))
        );
    }, [entries, search]);

    const dimsStr = (dims: number[]) => dims.map(d => `${d}D`).join(', ');

    const handleItemClick = (entry: GenomeLibraryEntry) => {
        if (multi) {
            setSelectedIds(prev => {
                const next = new Set(prev);
                if (next.has(entry.id)) next.delete(entry.id);
                else next.add(entry.id);
                return next;
            });
        } else {
            onSelect?.(entry);
        }
    };

    const handleConfirm = () => {
        const selected = entries.filter(e => selectedIds.has(e.id));
        onConfirm?.(selected);
    };

    return (
        <div className={styles.overlay} onClick={onClose}>
            <div className={styles.modal} onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className={styles.header}>
                    <h3 className={styles.title}>{title}</h3>
                    <button className={styles.closeBtn} onClick={onClose}>✕</button>
                </div>

                {/* Search */}
                <input
                    className={styles.searchInput}
                    placeholder="Search by name, tag, layer type..."
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    autoFocus
                />

                {/* List */}
                <div className={styles.list}>
                    {filtered.length === 0 ? (
                        <div className={styles.empty}>
                            {entries.length === 0 ? 'Library is empty' : 'No matches'}
                        </div>
                    ) : (
                        filtered.map(entry => (
                            <div
                                key={entry.id}
                                className={styles.item}
                                style={selectedIds.has(entry.id)
                                    ? { borderColor: 'var(--color-accent-primary)', background: 'rgba(99,102,241,0.1)' }
                                    : undefined
                                }
                                onClick={() => handleItemClick(entry)}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    {multi && (
                                        <input
                                            type="checkbox"
                                            checked={selectedIds.has(entry.id)}
                                            readOnly
                                            style={{ accentColor: 'var(--color-accent-primary)' }}
                                        />
                                    )}
                                    <div style={{ flex: 1 }}>
                                        <div className={styles.itemName}>{entry.name}</div>
                                        <div className={styles.itemMeta}>
                                            <span>{dimsStr(entry.inputDims)}→{dimsStr(entry.outputDims)}</span>
                                            <span>·</span>
                                            <span>{entry.totalNodes} nodes</span>
                                            {entry.layerTypes.map(lt => (
                                                <span key={lt} className={styles.layerBadge}>{lt}</span>
                                            ))}
                                        </div>
                                        {entry.tags.length > 0 && (
                                            <div className={styles.itemTags}>
                                                {entry.tags.map(t => (
                                                    <span key={t} className={styles.tag}>{t}</span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Confirm button (multi mode) */}
                {multi && (
                    <button
                        onClick={handleConfirm}
                        disabled={selectedIds.size === 0}
                        style={{
                            padding: '0.55rem 1rem',
                            borderRadius: '6px',
                            border: 'none',
                            background: selectedIds.size > 0 ? 'var(--color-accent-primary)' : 'var(--color-bg-tertiary)',
                            color: 'white',
                            fontWeight: 600,
                            cursor: selectedIds.size > 0 ? 'pointer' : 'not-allowed',
                            transition: 'background 0.15s',
                        }}
                    >
                        Confirm ({selectedIds.size} selected)
                    </button>
                )}
            </div>
        </div>
    );
};
