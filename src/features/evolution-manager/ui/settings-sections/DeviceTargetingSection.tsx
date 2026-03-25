import React, { useMemo } from 'react';
import type { DeviceLibraryImportMode, DeviceTemplateDto, UpdateDeviceTemplatePatch } from '../../../../shared/lib';
import { validateDeviceConstraintParams } from '../../model/deviceConstraints';
import { useDeviceLibrary } from '../../model/useDeviceLibrary';
import { useEvolutionSettingsStore } from '../../model/store';
import { DeviceLibraryManager } from '../DeviceLibraryManager';
import { DeviceProfileSelector, type SaveDeviceTemplatePayload } from '../DeviceProfileSelector';
import styles from './SettingsSections.module.css';

const MB_TO_BYTES = 1024 * 1024;

interface DeviceTargetingSectionProps {
    disabled?: boolean;
}

function toCustomDeviceParams(template: DeviceTemplateDto) {
    return {
        mops_budget: template.constraints.mops_budget,
        ram_mb: template.constraints.ram_budget_mb,
        flash_mb: template.constraints.flash_budget_mb,
        latency_budget_ms: template.constraints.max_latency_ms,
        max_model_size_mb: template.constraints.flash_budget_mb,
    };
}

export function DeviceTargetingSection({ disabled = false }: DeviceTargetingSectionProps) {
    const settings = useEvolutionSettingsStore();
    const deviceLibrary = useDeviceLibrary();

    const activeConstraints = useMemo(() => ({
        mops_budget: settings.customDeviceParams?.mops_budget ?? Math.max(1, settings.resourceTargets.macs / 1_000_000),
        ram_mb: settings.customDeviceParams?.ram_mb ?? Math.max(1, settings.resourceTargets.ram / MB_TO_BYTES),
        flash_mb:
            settings.customDeviceParams?.flash_mb ??
            settings.customDeviceParams?.max_model_size_mb ??
            Math.max(1, settings.resourceTargets.flash / MB_TO_BYTES),
        latency_budget_ms:
            settings.customDeviceParams?.latency_budget_ms ??
            settings.selectedDeviceProfile?.inference_latency_budget_ms ??
            50,
    }), [
        settings.customDeviceParams,
        settings.resourceTargets.flash,
        settings.resourceTargets.macs,
        settings.resourceTargets.ram,
        settings.selectedDeviceProfile?.inference_latency_budget_ms,
    ]);

    const validation = useMemo(() => validateDeviceConstraintParams(activeConstraints), [activeConstraints]);

    const estimatedParallelism = useMemo(() => {
        const perGenomeEstimateMb = Math.max(32, settings.memorySafetyMarginMb * 0.75);
        return Math.max(1, Math.floor(activeConstraints.ram_mb / perGenomeEstimateMb));
    }, [activeConstraints.ram_mb, settings.memorySafetyMarginMb]);

    const handleSaveAsTemplate = async (payload: SaveDeviceTemplatePayload) => {
        const created = await deviceLibrary.createTemplate({
            name: payload.name,
            notes: payload.notes,
            tags: payload.tags,
            constraints: {
                mops_budget: payload.constraints.mops_budget,
                ram_budget_mb: payload.constraints.ram_mb,
                flash_budget_mb: payload.constraints.flash_mb,
                max_latency_ms: payload.constraints.latency_budget_ms,
            },
        });

        settings.setIsCustomDevice(true);
        settings.setSelectedDeviceProfile({
            device_id: created.id,
            device_name: created.name,
            compute_capability: 'X86',
            ram_mb: Math.round(created.constraints.ram_budget_mb),
            inference_latency_budget_ms: created.constraints.max_latency_ms,
            training_available: true,
            max_model_size_mb: created.constraints.flash_budget_mb,
        });
        settings.setCustomDeviceParams(toCustomDeviceParams(created));
        settings.setResourceTarget('ram', Math.round(created.constraints.ram_budget_mb * MB_TO_BYTES));
        settings.setResourceTarget('flash', Math.round(created.constraints.flash_budget_mb * MB_TO_BYTES));
        settings.setResourceTarget('macs', Math.round(created.constraints.mops_budget * 1_000_000));
    };

    const handleApplyTemplate = (template: DeviceTemplateDto) => {
        settings.setIsCustomDevice(true);
        settings.setSelectedDeviceProfile({
            device_id: template.id,
            device_name: template.name,
            compute_capability: 'X86',
            ram_mb: Math.round(template.constraints.ram_budget_mb),
            inference_latency_budget_ms: template.constraints.max_latency_ms,
            training_available: true,
            max_model_size_mb: template.constraints.flash_budget_mb,
        });
        settings.setCustomDeviceParams(toCustomDeviceParams(template));
        settings.setResourceTarget('ram', Math.round(template.constraints.ram_budget_mb * MB_TO_BYTES));
        settings.setResourceTarget('flash', Math.round(template.constraints.flash_budget_mb * MB_TO_BYTES));
        settings.setResourceTarget('macs', Math.round(template.constraints.mops_budget * 1_000_000));
    };

    const handleUpdateTemplate = async (id: string, patch: UpdateDeviceTemplatePatch) => {
        await deviceLibrary.updateTemplate(id, patch);
    };

    const handleDuplicateTemplate = async (id: string, newName: string) => {
        await deviceLibrary.duplicateTemplate(id, newName);
    };

    const handleDeleteTemplate = async (id: string) => {
        await deviceLibrary.deleteTemplate(id);
    };

    const handleImportLibrary = async (path: string, mode: DeviceLibraryImportMode) => {
        await deviceLibrary.importLibrary(path, mode);
    };

    const handleExportLibrary = async (path: string) => {
        await deviceLibrary.exportLibrary(path);
    };

    return (
        <div className={styles.section}>
            <h4 className={styles.sectionTitle}>Device Targeting</h4>
            <DeviceProfileSelector disabled={disabled} onSaveAsTemplate={handleSaveAsTemplate} />

            <p className={styles.helper}>
                Estimated parallelism for selected constraints: {estimatedParallelism}
            </p>
            {!validation.isValid && (
                <p className={styles.error}>Some device constraints are invalid and may block feasible candidates.</p>
            )}
            {validation.warnings.length > 0 && (
                <p className={styles.warning}>
                    Constraints are strict. Feasible solutions may be limited.
                </p>
            )}

            <DeviceLibraryManager
                disabled={disabled}
                templates={deviceLibrary.templates}
                isLoading={deviceLibrary.isLoading}
                isMutating={deviceLibrary.isMutating}
                error={deviceLibrary.error}
                activeTemplateId={settings.selectedDeviceProfile?.device_id}
                lastImportCount={deviceLibrary.lastImportCount}
                lastExportCount={deviceLibrary.lastExportCount}
                onApplyTemplate={handleApplyTemplate}
                onUpdateTemplate={handleUpdateTemplate}
                onDuplicateTemplate={handleDuplicateTemplate}
                onDeleteTemplate={handleDeleteTemplate}
                onImportLibrary={handleImportLibrary}
                onExportLibrary={handleExportLibrary}
            />
        </div>
    );
}
