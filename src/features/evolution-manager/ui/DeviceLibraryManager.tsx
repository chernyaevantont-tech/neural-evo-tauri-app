import React, { useMemo, useState } from 'react';
import type {
    DeviceLibraryImportMode,
    DeviceTemplateDto,
    UpdateDeviceTemplatePatch,
} from '../../../shared/lib';
import styles from './DeviceLibraryManager.module.css';

type EditDraft = {
    name: string;
    notes: string;
    tags: string;
    mopsBudget: number;
    ramMb: number;
    flashMb: number;
    latencyMs: number;
};

export interface DeviceLibraryManagerProps {
    disabled?: boolean;
    templates: DeviceTemplateDto[];
    isLoading: boolean;
    isMutating: boolean;
    error: string | null;
    activeTemplateId?: string;
    lastImportCount: number | null;
    lastExportCount: number | null;
    onApplyTemplate: (template: DeviceTemplateDto) => void;
    onUpdateTemplate: (id: string, patch: UpdateDeviceTemplatePatch) => Promise<void> | void;
    onDuplicateTemplate: (id: string, newName: string) => Promise<void> | void;
    onDeleteTemplate: (id: string) => Promise<void> | void;
    onImportLibrary: (path: string, mode: DeviceLibraryImportMode) => Promise<void> | void;
    onExportLibrary: (path: string) => Promise<void> | void;
}

function formatUpdatedAt(unixMs: number): string {
    return new Date(unixMs).toLocaleString();
}

function toDraft(template: DeviceTemplateDto): EditDraft {
    return {
        name: template.name,
        notes: template.notes ?? '',
        tags: template.tags.join(', '),
        mopsBudget: template.constraints.mops_budget,
        ramMb: template.constraints.ram_budget_mb,
        flashMb: template.constraints.flash_budget_mb,
        latencyMs: template.constraints.max_latency_ms,
    };
}

export function DeviceLibraryManager({
    disabled = false,
    templates,
    isLoading,
    isMutating,
    error,
    activeTemplateId,
    lastImportCount,
    lastExportCount,
    onApplyTemplate,
    onUpdateTemplate,
    onDuplicateTemplate,
    onDeleteTemplate,
    onImportLibrary,
    onExportLibrary,
}: DeviceLibraryManagerProps) {
    const [editingTemplateId, setEditingTemplateId] = useState<string | null>(null);
    const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
    const [importPath, setImportPath] = useState<string>('');
    const [exportPath, setExportPath] = useState<string>('');
    const [importMode, setImportMode] = useState<DeviceLibraryImportMode>('merge');

    const activeTemplateName = useMemo(() => {
        if (!activeTemplateId) {
            return null;
        }
        return templates.find((template) => template.id === activeTemplateId)?.name ?? null;
    }, [activeTemplateId, templates]);

    const startEdit = (template: DeviceTemplateDto) => {
        setEditingTemplateId(template.id);
        setEditDraft(toDraft(template));
    };

    const cancelEdit = () => {
        setEditingTemplateId(null);
        setEditDraft(null);
    };

    const saveEdit = async () => {
        if (!editingTemplateId || !editDraft) {
            return;
        }

        const patch: UpdateDeviceTemplatePatch = {
            name: editDraft.name.trim(),
            notes: editDraft.notes.trim() || undefined,
            tags: editDraft.tags
                .split(',')
                .map((tag) => tag.trim())
                .filter(Boolean),
            constraints: {
                mops_budget: Number(editDraft.mopsBudget),
                ram_budget_mb: Number(editDraft.ramMb),
                flash_budget_mb: Number(editDraft.flashMb),
                max_latency_ms: Number(editDraft.latencyMs),
            },
        };

        await onUpdateTemplate(editingTemplateId, patch);
        cancelEdit();
    };

    const canMutate = !disabled && !isMutating;

    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <p className={styles.title}>Device Library</p>
                {activeTemplateName && (
                    <span className={styles.activeBadge} aria-label="Active device template">
                        Active: {activeTemplateName}
                    </span>
                )}
            </div>

            <div className={styles.toolbar}>
                <input
                    value={importPath}
                    onChange={(event) => setImportPath(event.target.value)}
                    placeholder="Import file path"
                    className={styles.input}
                    disabled={disabled}
                    aria-label="Import file path"
                />
                <select
                    value={importMode}
                    onChange={(event) => setImportMode(event.target.value as DeviceLibraryImportMode)}
                    className={styles.select}
                    disabled={disabled}
                    aria-label="Import mode"
                >
                    <option value="merge">Merge</option>
                    <option value="replace">Replace</option>
                </select>
                <button
                    type="button"
                    className={styles.button}
                    disabled={disabled || !importPath.trim() || isMutating}
                    onClick={() => onImportLibrary(importPath.trim(), importMode)}
                >
                    Import library
                </button>
            </div>

            <div className={styles.toolbar}>
                <input
                    value={exportPath}
                    onChange={(event) => setExportPath(event.target.value)}
                    placeholder="Export file path"
                    className={styles.input}
                    disabled={disabled}
                    aria-label="Export file path"
                />
                <button
                    type="button"
                    className={styles.button}
                    disabled={disabled || !exportPath.trim() || isMutating}
                    onClick={() => onExportLibrary(exportPath.trim())}
                >
                    Export library
                </button>
            </div>

            {lastImportCount !== null && (
                <div className={styles.info}>Import completed: {lastImportCount} templates.</div>
            )}
            {lastExportCount !== null && (
                <div className={styles.info}>Export completed: {lastExportCount} templates.</div>
            )}
            {error && <div className={styles.error}>{error}</div>}

            <div className={styles.tableWrap}>
                {isLoading ? (
                    <div className={styles.info}>Loading device templates...</div>
                ) : (
                    <table className={styles.table}>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>MOPS</th>
                                <th>RAM MB</th>
                                <th>FLASH MB</th>
                                <th>Max latency ms</th>
                                <th>Updated at</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {templates.length === 0 && (
                                <tr>
                                    <td colSpan={7} className={styles.empty}>No templates saved yet.</td>
                                </tr>
                            )}
                            {templates.map((template) => {
                                const isActive = activeTemplateId === template.id;
                                const isEditing = editingTemplateId === template.id && editDraft;

                                return (
                                    <React.Fragment key={template.id}>
                                        <tr className={isActive ? styles.activeRow : undefined}>
                                            <td>{template.name}</td>
                                            <td>{template.constraints.mops_budget}</td>
                                            <td>{template.constraints.ram_budget_mb}</td>
                                            <td>{template.constraints.flash_budget_mb}</td>
                                            <td>{template.constraints.max_latency_ms}</td>
                                            <td>{formatUpdatedAt(template.updated_at_unix_ms)}</td>
                                            <td>
                                                <div className={styles.actionRow}>
                                                    <button
                                                        type="button"
                                                        className={styles.button}
                                                        onClick={() => onApplyTemplate(template)}
                                                        disabled={disabled}
                                                    >
                                                        Apply
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={styles.button}
                                                        onClick={() => startEdit(template)}
                                                        disabled={!canMutate}
                                                    >
                                                        Edit
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={styles.button}
                                                        onClick={() => onDuplicateTemplate(template.id, `${template.name} Copy`)}
                                                        disabled={!canMutate}
                                                    >
                                                        Duplicate
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={styles.buttonDanger}
                                                        onClick={() => onDeleteTemplate(template.id)}
                                                        disabled={!canMutate}
                                                    >
                                                        Delete
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                        {isEditing && (
                                            <tr className={styles.editRow}>
                                                <td colSpan={7}>
                                                    <div className={styles.editGrid}>
                                                        <label className={styles.editField}>
                                                            <span>Name</span>
                                                            <input
                                                                className={styles.input}
                                                                value={editDraft.name}
                                                                onChange={(event) => setEditDraft({ ...editDraft, name: event.target.value })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>Notes</span>
                                                            <input
                                                                className={styles.input}
                                                                value={editDraft.notes}
                                                                onChange={(event) => setEditDraft({ ...editDraft, notes: event.target.value })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>Tags (comma separated)</span>
                                                            <input
                                                                className={styles.input}
                                                                value={editDraft.tags}
                                                                onChange={(event) => setEditDraft({ ...editDraft, tags: event.target.value })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>MOPS</span>
                                                            <input
                                                                className={styles.input}
                                                                type="number"
                                                                value={editDraft.mopsBudget}
                                                                onChange={(event) => setEditDraft({ ...editDraft, mopsBudget: Number(event.target.value) })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>RAM MB</span>
                                                            <input
                                                                className={styles.input}
                                                                type="number"
                                                                value={editDraft.ramMb}
                                                                onChange={(event) => setEditDraft({ ...editDraft, ramMb: Number(event.target.value) })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>FLASH MB</span>
                                                            <input
                                                                className={styles.input}
                                                                type="number"
                                                                value={editDraft.flashMb}
                                                                onChange={(event) => setEditDraft({ ...editDraft, flashMb: Number(event.target.value) })}
                                                            />
                                                        </label>
                                                        <label className={styles.editField}>
                                                            <span>Max latency ms</span>
                                                            <input
                                                                className={styles.input}
                                                                type="number"
                                                                value={editDraft.latencyMs}
                                                                onChange={(event) => setEditDraft({ ...editDraft, latencyMs: Number(event.target.value) })}
                                                            />
                                                        </label>
                                                    </div>
                                                    <div className={styles.actionRow}>
                                                        <button
                                                            type="button"
                                                            className={styles.button}
                                                            disabled={!canMutate || !editDraft.name.trim()}
                                                            onClick={saveEdit}
                                                        >
                                                            Save changes
                                                        </button>
                                                        <button type="button" className={styles.button} onClick={cancelEdit}>
                                                            Cancel
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
