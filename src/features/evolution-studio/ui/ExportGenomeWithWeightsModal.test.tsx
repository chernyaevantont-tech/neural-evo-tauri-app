import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ExportGenomeWithWeightsModal } from './ExportGenomeWithWeightsModal';

const invokeMock = vi.fn();

vi.mock('@tauri-apps/api/core', () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

describe('ExportGenomeWithWeightsModal', () => {
    it('exports successfully and shows result paths', async () => {
        invokeMock.mockImplementation((cmd: string) => {
            if (cmd === 'has_cached_weights') {
                return Promise.resolve(true);
            }
            if (cmd === 'export_genome_with_weights') {
                return Promise.resolve({
                    weights_path: 'C:/tmp/g-1.mpk',
                    metadata_path: 'C:/tmp/metadata.json',
                    used_cached_weights: true,
                });
            }
            return Promise.resolve('');
        });

        render(<ExportGenomeWithWeightsModal genomeId="g-1" onClose={() => {}} />);

        const input = await screen.findByPlaceholderText('Select target directory');
        fireEvent.change(input, { target: { value: 'C:/tmp' } });

        fireEvent.click(screen.getByRole('button', { name: 'Export' }));

        await waitFor(() => {
            expect(screen.getByText(/Export complete/i)).toBeTruthy();
            expect(screen.getByText(/C:\/tmp\/g-1\.mpk/i)).toBeTruthy();
            expect(screen.getByText(/C:\/tmp\/metadata\.json/i)).toBeTruthy();
        });
    });

    it('shows error state on invalid export path', async () => {
        invokeMock.mockImplementation((cmd: string) => {
            if (cmd === 'has_cached_weights') {
                return Promise.resolve(false);
            }
            if (cmd === 'export_genome_with_weights') {
                return Promise.reject(new Error('invalid path'));
            }
            return Promise.resolve('');
        });

        render(<ExportGenomeWithWeightsModal genomeId="g-1" onClose={() => {}} />);

        const input = await screen.findByPlaceholderText('Select target directory');
        fireEvent.change(input, { target: { value: 'Z:/bad/path' } });

        const exportBtn = screen.getByRole('button', { name: 'Export' });
        fireEvent.click(exportBtn);

        await waitFor(() => {
            expect(screen.getByText(/invalid path/i)).toBeTruthy();
        });
    });
});
