import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { DeviceTemplateDto } from '../../../shared/lib';
import { DeviceLibraryManager } from './DeviceLibraryManager';

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

describe('DeviceLibraryManager', () => {
    it('applies template in one click', () => {
        const onApplyTemplate = vi.fn();

        render(
            <DeviceLibraryManager
                templates={[template()]}
                isLoading={false}
                isMutating={false}
                error={null}
                activeTemplateId={undefined}
                lastImportCount={null}
                lastExportCount={null}
                onApplyTemplate={onApplyTemplate}
                onUpdateTemplate={vi.fn()}
                onDuplicateTemplate={vi.fn()}
                onDeleteTemplate={vi.fn()}
                onImportLibrary={vi.fn()}
                onExportLibrary={vi.fn()}
            />,
        );

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

        expect(onApplyTemplate).toHaveBeenCalledWith(expect.objectContaining({ id: 'tpl-1' }));
    });

    it('calls import/export actions with selected paths and mode', () => {
        const onImportLibrary = vi.fn();
        const onExportLibrary = vi.fn();

        render(
            <DeviceLibraryManager
                templates={[template()]}
                isLoading={false}
                isMutating={false}
                error={null}
                activeTemplateId={undefined}
                lastImportCount={null}
                lastExportCount={null}
                onApplyTemplate={vi.fn()}
                onUpdateTemplate={vi.fn()}
                onDuplicateTemplate={vi.fn()}
                onDeleteTemplate={vi.fn()}
                onImportLibrary={onImportLibrary}
                onExportLibrary={onExportLibrary}
            />,
        );

        fireEvent.change(screen.getByLabelText('Import file path'), { target: { value: 'C:/tmp/in.json' } });
        fireEvent.change(screen.getByLabelText('Import mode'), { target: { value: 'replace' } });
        fireEvent.click(screen.getByRole('button', { name: 'Import library' }));

        fireEvent.change(screen.getByLabelText('Export file path'), { target: { value: 'C:/tmp/out.json' } });
        fireEvent.click(screen.getByRole('button', { name: 'Export library' }));

        expect(onImportLibrary).toHaveBeenCalledWith('C:/tmp/in.json', 'replace');
        expect(onExportLibrary).toHaveBeenCalledWith('C:/tmp/out.json');
    });
});
