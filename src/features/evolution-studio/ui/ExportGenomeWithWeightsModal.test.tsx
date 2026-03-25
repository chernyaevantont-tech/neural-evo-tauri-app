import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ExportGenomeWithWeightsModal } from './ExportGenomeWithWeightsModal';

const invokeMock = vi.fn();

vi.mock('@tauri-apps/api/core', () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

describe('ExportGenomeWithWeightsModal', () => {
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
