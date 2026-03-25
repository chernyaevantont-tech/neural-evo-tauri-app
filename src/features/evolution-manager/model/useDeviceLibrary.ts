import { invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useState } from 'react';
import type {
    CreateDeviceTemplateInput,
    DeviceLibraryImportMode,
    DeviceTemplateDto,
    UpdateDeviceTemplatePatch,
} from '../../../shared/lib';

function toErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

function sortByUpdatedAtDesc(templates: DeviceTemplateDto[]): DeviceTemplateDto[] {
    return [...templates].sort((a, b) => b.updated_at_unix_ms - a.updated_at_unix_ms);
}

export function useDeviceLibrary() {
    const [templates, setTemplates] = useState<DeviceTemplateDto[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [isMutating, setIsMutating] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [lastImportCount, setLastImportCount] = useState<number | null>(null);
    const [lastExportCount, setLastExportCount] = useState<number | null>(null);

    const refreshTemplates = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const result = await invoke<DeviceTemplateDto[]>('list_device_templates');
            setTemplates(sortByUpdatedAtDesc(result));
        } catch (err) {
            setError(toErrorMessage(err));
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        refreshTemplates().catch(() => {
            // Error is already handled in refreshTemplates.
        });
    }, [refreshTemplates]);

    const createTemplate = useCallback(async (input: CreateDeviceTemplateInput) => {
        setIsMutating(true);
        setError(null);
        try {
            const created = await invoke<DeviceTemplateDto>('create_device_template', { input });
            setTemplates((prev) => sortByUpdatedAtDesc([...prev, created]));
            return created;
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    const updateTemplate = useCallback(async (id: string, patch: UpdateDeviceTemplatePatch) => {
        setIsMutating(true);
        setError(null);
        try {
            const updated = await invoke<DeviceTemplateDto>('update_device_template', { id, patch });
            setTemplates((prev) => {
                const next = prev.map((template) => (template.id === id ? updated : template));
                return sortByUpdatedAtDesc(next);
            });
            return updated;
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    const deleteTemplate = useCallback(async (id: string) => {
        setIsMutating(true);
        setError(null);
        try {
            await invoke<void>('delete_device_template', { id });
            setTemplates((prev) => prev.filter((template) => template.id !== id));
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    const duplicateTemplate = useCallback(async (id: string, newName: string) => {
        setIsMutating(true);
        setError(null);
        try {
            const duplicated = await invoke<DeviceTemplateDto>('duplicate_device_template', {
                id,
                newName,
            });
            setTemplates((prev) => sortByUpdatedAtDesc([...prev, duplicated]));
            return duplicated;
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    const exportLibrary = useCallback(async (path: string) => {
        setIsMutating(true);
        setError(null);
        try {
            const exportedCount = await invoke<number>('export_device_library', { path });
            setLastExportCount(exportedCount);
            return exportedCount;
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    const importLibrary = useCallback(async (path: string, mode: DeviceLibraryImportMode) => {
        setIsMutating(true);
        setError(null);
        try {
            const imported = await invoke<DeviceTemplateDto[]>('import_device_library', { path, mode });
            const sorted = sortByUpdatedAtDesc(imported);
            setTemplates(sorted);
            setLastImportCount(sorted.length);
            return sorted;
        } catch (err) {
            const message = toErrorMessage(err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsMutating(false);
        }
    }, []);

    return {
        templates,
        isLoading,
        isMutating,
        error,
        lastImportCount,
        lastExportCount,
        refreshTemplates,
        createTemplate,
        updateTemplate,
        deleteTemplate,
        duplicateTemplate,
        exportLibrary,
        importLibrary,
    };
}
