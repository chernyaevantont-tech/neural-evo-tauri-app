import React, { useEffect, useMemo, useState } from 'react';
import type { DeviceProfile } from '../../../shared/lib';
import { useEvolutionSettingsStore } from '../model/store';
import { validateDeviceConstraintParams, type DeviceConstraintParams } from '../model/deviceConstraints';
import styles from './DeviceProfileSelector.module.css';

export type SaveDeviceTemplatePayload = {
    name: string;
    notes?: string;
    tags: string[];
    constraints: DeviceConstraintParams;
};

type DeviceTemplateCallbacks = {
    onSaveAsTemplate?: (payload: SaveDeviceTemplatePayload) => void;
};

type BuiltInProfile = {
    id: string;
    name: string;
    constraints: DeviceConstraintParams;
};

const MB_TO_BYTES = 1024 * 1024;

const BUILT_IN_PROFILES: BuiltInProfile[] = [
    {
        id: 'embedded-mcu',
        name: 'Embedded MCU',
        constraints: { mops_budget: 120, ram_mb: 0.5, flash_mb: 2, latency_budget_ms: 80 },
    },
    {
        id: 'edge-tiny',
        name: 'Edge Tiny',
        constraints: { mops_budget: 800, ram_mb: 8, flash_mb: 32, latency_budget_ms: 50 },
    },
    {
        id: 'mobile-low-end',
        name: 'Mobile Low-End',
        constraints: { mops_budget: 3000, ram_mb: 128, flash_mb: 256, latency_budget_ms: 45 },
    },
    {
        id: 'mobile-mid-range',
        name: 'Mobile Mid-Range',
        constraints: { mops_budget: 8000, ram_mb: 256, flash_mb: 512, latency_budget_ms: 30 },
    },
    {
        id: 'laptop-cpu',
        name: 'Laptop CPU',
        constraints: { mops_budget: 25000, ram_mb: 2048, flash_mb: 2048, latency_budget_ms: 25 },
    },
];

function toDeviceProfile(profile: BuiltInProfile): DeviceProfile {
    return {
        device_id: profile.id,
        device_name: profile.name,
        compute_capability: 'X86',
        ram_mb: Math.round(profile.constraints.ram_mb),
        inference_latency_budget_ms: profile.constraints.latency_budget_ms,
        training_available: true,
        max_model_size_mb: profile.constraints.flash_mb,
    };
}

function toCustomFromConstraints(constraints: DeviceConstraintParams) {
    return {
        mops_budget: constraints.mops_budget,
        ram_mb: constraints.ram_mb,
        flash_mb: constraints.flash_mb,
        latency_budget_ms: constraints.latency_budget_ms,
        max_model_size_mb: constraints.flash_mb,
    };
}

export function DeviceProfileSelector({
    disabled = false,
    onSaveAsTemplate,
}: { disabled?: boolean } & DeviceTemplateCallbacks) {
    const settings = useEvolutionSettingsStore();
    const [showSaveTemplateForm, setShowSaveTemplateForm] = useState(false);
    const [templateName, setTemplateName] = useState('');
    const [templateNotes, setTemplateNotes] = useState('');
    const [templateTags, setTemplateTags] = useState('');

    const activeConstraints = useMemo<DeviceConstraintParams>(() => {
        const selectedProfile = BUILT_IN_PROFILES.find((profile) => profile.id === settings.deviceProfileId);
        const fallback = selectedProfile?.constraints ?? BUILT_IN_PROFILES[0].constraints;
        return {
            mops_budget: settings.customDeviceParams?.mops_budget ?? fallback.mops_budget,
            ram_mb: settings.customDeviceParams?.ram_mb ?? fallback.ram_mb,
            flash_mb:
                settings.customDeviceParams?.flash_mb ??
                settings.customDeviceParams?.max_model_size_mb ??
                fallback.flash_mb,
            latency_budget_ms: settings.customDeviceParams?.latency_budget_ms ?? fallback.latency_budget_ms,
        };
    }, [settings.customDeviceParams, settings.deviceProfileId]);

    const validation = useMemo(() => validateDeviceConstraintParams(activeConstraints), [activeConstraints]);

    useEffect(() => {
        if (settings.selectedDeviceProfile) {
            return;
        }
        const defaultProfile = BUILT_IN_PROFILES[0];
        settings.setDeviceProfileId(defaultProfile.id);
        settings.setSelectedDeviceProfile(toDeviceProfile(defaultProfile));
        settings.setCustomDeviceParams(toCustomFromConstraints(defaultProfile.constraints));
        settings.setResourceTarget('ram', Math.round(defaultProfile.constraints.ram_mb * MB_TO_BYTES));
        settings.setResourceTarget('flash', Math.round(defaultProfile.constraints.flash_mb * MB_TO_BYTES));
        settings.setResourceTarget('macs', Math.round(defaultProfile.constraints.mops_budget * 1_000_000));
    }, [settings]);

    const setConstraint = (key: keyof DeviceConstraintParams, value: number) => {
        const next = { ...activeConstraints, [key]: value };
        settings.setCustomDeviceParams(toCustomFromConstraints(next));

        if (key === 'ram_mb') {
            settings.setResourceTarget('ram', Math.round(Math.max(0, value) * MB_TO_BYTES));
        }
        if (key === 'flash_mb') {
            settings.setResourceTarget('flash', Math.round(Math.max(0, value) * MB_TO_BYTES));
        }
        if (key === 'mops_budget') {
            settings.setResourceTarget('macs', Math.round(Math.max(0, value) * 1_000_000));
        }
    };

    const handleSelectProfile = (profileId: string) => {
        const profile = BUILT_IN_PROFILES.find((item) => item.id === profileId);
        if (!profile) {
            return;
        }

        settings.setDeviceProfileId(profile.id);
        settings.setIsCustomDevice(false);
        settings.setSelectedDeviceProfile(toDeviceProfile(profile));
        settings.setCustomDeviceParams(toCustomFromConstraints(profile.constraints));
        settings.setResourceTarget('ram', Math.round(profile.constraints.ram_mb * MB_TO_BYTES));
        settings.setResourceTarget('flash', Math.round(profile.constraints.flash_mb * MB_TO_BYTES));
        settings.setResourceTarget('macs', Math.round(profile.constraints.mops_budget * 1_000_000));
    };

    const saveTemplate = () => {
        const trimmed = templateName.trim();
        if (!trimmed || disabled) {
            return;
        }

        const payload: SaveDeviceTemplatePayload = {
            name: trimmed,
            notes: templateNotes.trim() || undefined,
            tags: templateTags
                .split(',')
                .map((tag) => tag.trim())
                .filter(Boolean),
            constraints: activeConstraints,
        };

        if (onSaveAsTemplate) {
            onSaveAsTemplate(payload);
        } else {
            console.info('TODO(T116): save device template', payload);
        }

        setTemplateName('');
        setTemplateNotes('');
        setTemplateTags('');
        setShowSaveTemplateForm(false);
    };

    return (
        <div className={styles.container}>
            <div className={styles.row}>
                <label className={styles.label} htmlFor="built-in-profile">Built-in profile</label>
            </div>
            <select
                id="built-in-profile"
                className={styles.select}
                value={settings.deviceProfileId}
                onChange={(event) => handleSelectProfile(event.target.value)}
                disabled={disabled || settings.isCustomDevice}
                aria-label="Built-in profile"
            >
                {BUILT_IN_PROFILES.map((profile) => (
                    <option key={profile.id} value={profile.id}>
                        {profile.name}
                    </option>
                ))}
            </select>

            <label className={styles.row}>
                <span className={styles.label}>Custom constraints</span>
                <input
                    type="checkbox"
                    checked={settings.isCustomDevice}
                    onChange={(event) => settings.setIsCustomDevice(event.target.checked)}
                    disabled={disabled}
                    aria-label="Custom constraints"
                />
            </label>

            <div className={styles.fields}>
                <label className={styles.field}>
                    <span className={styles.label}>MOPS budget</span>
                    <input
                        className={styles.input}
                        type="number"
                        min="0"
                        step="0.1"
                        value={activeConstraints.mops_budget}
                        onChange={(event) => setConstraint('mops_budget', Number(event.target.value))}
                        disabled={disabled || !settings.isCustomDevice}
                        aria-label="MOPS budget"
                    />
                </label>
                {validation.fieldErrors.mops_budget && (
                    <div className={styles.fieldError}>{validation.fieldErrors.mops_budget}</div>
                )}

                <label className={styles.field}>
                    <span className={styles.label}>RAM budget (MB)</span>
                    <input
                        className={styles.input}
                        type="number"
                        min="0"
                        step="0.1"
                        value={activeConstraints.ram_mb}
                        onChange={(event) => setConstraint('ram_mb', Number(event.target.value))}
                        disabled={disabled || !settings.isCustomDevice}
                        aria-label="RAM budget (MB)"
                    />
                </label>
                {validation.fieldErrors.ram_mb && (
                    <div className={styles.fieldError}>{validation.fieldErrors.ram_mb}</div>
                )}

                <label className={styles.field}>
                    <span className={styles.label}>FLASH budget (MB)</span>
                    <input
                        className={styles.input}
                        type="number"
                        min="0"
                        step="0.1"
                        value={activeConstraints.flash_mb}
                        onChange={(event) => setConstraint('flash_mb', Number(event.target.value))}
                        disabled={disabled || !settings.isCustomDevice}
                        aria-label="FLASH budget (MB)"
                    />
                </label>
                {validation.fieldErrors.flash_mb && (
                    <div className={styles.fieldError}>{validation.fieldErrors.flash_mb}</div>
                )}

                <label className={styles.field}>
                    <span className={styles.label}>Max latency (ms)</span>
                    <input
                        className={styles.input}
                        type="number"
                        min="0"
                        step="0.1"
                        value={activeConstraints.latency_budget_ms}
                        onChange={(event) => setConstraint('latency_budget_ms', Number(event.target.value))}
                        disabled={disabled || !settings.isCustomDevice}
                        aria-label="Max latency (ms)"
                    />
                </label>
                {validation.fieldErrors.latency_budget_ms && (
                    <div className={styles.fieldError}>{validation.fieldErrors.latency_budget_ms}</div>
                )}

                {validation.warnings.map((warning) => (
                    <div key={warning} className={styles.warning}>{warning}</div>
                ))}
            </div>

            <div className={styles.feasibleFilter}>
                <label className={styles.row}>
                    <span className={styles.label}>Show only feasible</span>
                    <input
                        type="checkbox"
                        checked={settings.showOnlyFeasible}
                        onChange={(event) => settings.setShowOnlyFeasible(event.target.checked)}
                        disabled={disabled}
                        aria-label="Show only feasible"
                    />
                </label>
            </div>

            <div className={styles.templates}>
                <p className={styles.templateTitle}>Device templates</p>
                <button
                    type="button"
                    className={styles.button}
                    onClick={() => setShowSaveTemplateForm((prev) => !prev)}
                    disabled={disabled}
                >
                    Save as device template
                </button>

                {showSaveTemplateForm && (
                    <div className={styles.templateForm}>
                        <input
                            className={styles.templateInput}
                            value={templateName}
                            onChange={(event) => setTemplateName(event.target.value)}
                            placeholder="Template name"
                            disabled={disabled}
                            aria-label="Template name"
                        />
                        <input
                            className={styles.templateInput}
                            value={templateNotes}
                            onChange={(event) => setTemplateNotes(event.target.value)}
                            placeholder="Notes (optional)"
                            disabled={disabled}
                            aria-label="Template notes"
                        />
                        <input
                            className={styles.templateInput}
                            value={templateTags}
                            onChange={(event) => setTemplateTags(event.target.value)}
                            placeholder="Tags (comma separated)"
                            disabled={disabled}
                            aria-label="Template tags"
                        />
                        <div className={styles.templateActions}>
                            <button
                                type="button"
                                className={styles.button}
                                onClick={saveTemplate}
                                disabled={disabled || !templateName.trim()}
                            >
                                Save template
                            </button>
                            <button
                                type="button"
                                className={styles.button}
                                onClick={() => setShowSaveTemplateForm(false)}
                                disabled={disabled}
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
