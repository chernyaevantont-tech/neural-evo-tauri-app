import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { DeviceTemplateDto } from '../../../shared/lib';
import { useDeviceLibrary } from './useDeviceLibrary';

const invokeMock = vi.fn();

vi.mock('@tauri-apps/api/core', () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

function template(overrides?: Partial<DeviceTemplateDto>): DeviceTemplateDto {
    return {
        id: 'tpl-1',
        name: 'Edge Tiny',
        constraints: {
            mops_budget: 800,
            ram_budget_mb: 8,
            flash_budget_mb: 32,
            max_latency_ms: 50,
        },
        notes: 'baseline',
        tags: ['edge'],
        created_at_unix_ms: 100,
        updated_at_unix_ms: 100,
        ...overrides,
    };
}

describe('useDeviceLibrary', () => {
    beforeEach(() => {
        invokeMock.mockReset();
    });

    it('loads templates on mount (list flow)', async () => {
        invokeMock.mockImplementation((command: string) => {
            if (command === 'list_device_templates') {
                return Promise.resolve([template()]);
            }
            return Promise.resolve(null);
        });

        const { result } = renderHook(() => useDeviceLibrary());

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.templates).toHaveLength(1);
        expect(result.current.templates[0].name).toBe('Edge Tiny');
    });

    it('supports create/update/delete template flows', async () => {
        invokeMock.mockImplementation((command: string, args?: Record<string, unknown>) => {
            if (command === 'list_device_templates') {
                return Promise.resolve([]);
            }
            if (command === 'create_device_template') {
                const input = args?.input as { name: string } | undefined;
                return Promise.resolve(template({ id: 'tpl-new', name: input?.name ?? 'Edge Tiny' }));
            }
            if (command === 'update_device_template') {
                return Promise.resolve(template({ id: args?.id as string, name: 'Edge Tiny v2', updated_at_unix_ms: 200 }));
            }
            if (command === 'delete_device_template') {
                return Promise.resolve(null);
            }
            return Promise.resolve(null);
        });

        const { result } = renderHook(() => useDeviceLibrary());
        await waitFor(() => expect(result.current.isLoading).toBe(false));

        await act(async () => {
            await result.current.createTemplate({
                name: 'Edge Tiny',
                notes: 'new',
                tags: ['edge'],
                constraints: {
                    mops_budget: 800,
                    ram_budget_mb: 8,
                    flash_budget_mb: 32,
                    max_latency_ms: 50,
                },
            });
        });
        expect(result.current.templates).toHaveLength(1);

        await act(async () => {
            await result.current.updateTemplate('tpl-new', { name: 'Edge Tiny v2' });
        });
        expect(result.current.templates[0].name).toBe('Edge Tiny v2');

        await act(async () => {
            await result.current.deleteTemplate('tpl-new');
        });
        expect(result.current.templates).toHaveLength(0);
    });

    it('stores API errors and rethrows', async () => {
        invokeMock.mockImplementation((command: string) => {
            if (command === 'list_device_templates') {
                return Promise.resolve([]);
            }
            if (command === 'create_device_template') {
                return Promise.reject(new Error('duplicate name'));
            }
            return Promise.resolve(null);
        });

        const { result } = renderHook(() => useDeviceLibrary());
        await waitFor(() => expect(result.current.isLoading).toBe(false));

        await act(async () => {
            await expect(
                result.current.createTemplate({
                    name: 'Edge Tiny',
                    notes: undefined,
                    tags: [],
                    constraints: {
                        mops_budget: 1,
                        ram_budget_mb: 1,
                        flash_budget_mb: 1,
                        max_latency_ms: 1,
                    },
                }),
            ).rejects.toThrow('duplicate name');
        });

        await waitFor(() => {
            expect(result.current.error).toContain('duplicate name');
        });
    });
});
